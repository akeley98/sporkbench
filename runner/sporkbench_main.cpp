#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <errno.h>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <string.h>
#include <tuple>
#include <vector>

#include "sporkbench_cases.hpp"
#include "sporkbench_pcg3d.hpp"
#include "sporkbench_test.hpp"

#define CUBLAS_CHECK(x) if (auto _cublas_status = x; _cublas_status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%i cublas status %i\n", __FILE__, __LINE__, (int)_cublas_status); }

namespace sporkbench
{

bool global_all_passed = true;
constexpr int num_warmup = 4;
constexpr int num_timed = 100;
constexpr size_t L2_shred_bytes = 1u << 27;

struct MainData
{
    int cuda_cc_major;
    int cuda_cc_minor;
    cudaStream_t stream;
    cublasHandle_t cublasH;
    cudaEvent_t start_event, end_event;
    FILE* json_file;
};

struct TestDataConfig
{
    TestDataCode A_code;
    TestDataCode B_code;
    TestCheckMode check_mode;
};

struct GemmPlotSize
{
    int L, M, N, K;
};

struct GemmPlotInput
{
    std::string name;
    std::string title;
    std::string x_axis;
    std::vector<GemmPlotSize> sizes;
};

struct GemvPlotSize
{
    int M, K;
};

struct GemvPlotInput
{
    std::string name;
    std::string title;
    std::string x_axis;
    std::vector<GemvPlotSize> sizes;
};

std::vector<GemmPlotInput> generate_gemm_plot_inputs(bool is_h100)
{
    std::vector<GemmPlotInput> plots;

    auto add_MNK = [] (int M, int N, int K, GemmPlotInput& non_batched, GemmPlotInput& batched)
    {
        non_batched.sizes.push_back(GemmPlotSize{1, M, N, K});
        batched.sizes.push_back(GemmPlotSize{4, M, N, K});
    };

    if (is_h100) {
        GemmPlotInput L1K512{"L1K512", "GEMM, non-batched, N=1536, K=512", "M", {}};
        GemmPlotInput L4K512{"L4K512", "GEMM, batched, L=4, N=1536, K=512", "M", {}};
        GemmPlotInput L1K65536{"L1K65536", "GEMM, non-batched, N=1536, K=65536", "M", {}};
        for (int M = 256; M <= 4096; M += 256) {
            const int N = 1536;
            add_MNK(M, N, 512, L1K512, L4K512);
            L1K65536.sizes.push_back({1, M, N, 65536});
        }
        plots.push_back(L1K512);
        plots.push_back(L4K512);
        plots.push_back(L1K65536);
    }
    GemmPlotInput L1_square{"L1_square", "GEMM, non-batched, M=N=K", "M", {}};
    GemmPlotInput L4_square{"L4_square", "GEMM, batched, L=4, M=N=K", "M", {}};
    for (int M = 512; M <= 4096; M += 512) {
        add_MNK(M, M, M, L1_square, L4_square);
    }
    if (is_h100) {
        for (int M = 2048 * 3; M <= 2048 * 6; M += 2048) {
            L1_square.sizes.push_back(GemmPlotSize{1, M, M, M});
        }
    }
    plots.push_back(L1_square);
    plots.push_back(L4_square);
    return plots;
}

std::vector<GemvPlotInput> generate_gemv_plot_inputs()
{
    GemvPlotInput plot_input{};
    plot_input.name = "gemv";
    plot_input.title = "GEMV, M=K";
    plot_input.x_axis = "M";
    for (int m = 1024; m <= 8192; m *= 2) {
        plot_input.sizes.push_back({m, m});
    }
    return {plot_input};

}

static void no_op_warn_if_no_json()
{

}

void (*warn_if_no_json)() = no_op_warn_if_no_json;

struct AsyncDeleter
{
    cudaStream_t stream;

    void operator() (void* victim)
    {
        cudaFreeAsync(victim, stream);
    }
};

std::unique_ptr<float[], AsyncDeleter> alloc_f32(int L, int MN, int K, AsyncDeleter deleter)
{
    size_t sz = sizeof(float) * size_t(L) * size_t(MN) * size_t(K);
    void* ptr = nullptr;
    cudaMallocAsync(&ptr, sz, deleter.stream);
    if (!ptr and sz > 0) {
        cudaError_t err = cudaGetLastError();
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return std::unique_ptr<float[], AsyncDeleter>(static_cast<float*>(ptr), deleter);
}

template <typename KernelCase>
struct KernelCaseEntry
{
    const KernelCase* p_case;
    bool is_builtin;
    std::vector<double> flops_samples;
    int K_split;
};

void shuffle_ints(std::vector<int>& ints, int y, int z)
{
    auto cmp = [y, z] (int x1, int x2)
    {
        return pcg3d(uint32_t(x1), uint32_t(y), uint32_t(z)) < pcg3d(uint32_t(x2), uint32_t(y), uint32_t(z));
    };
    std::sort(ints.begin(), ints.end(), cmp);
}

template <typename KernelCase>
std::vector<KernelCaseEntry<KernelCase>> generate_cases(
    std::vector<int>* p_case_permutations, int cuda_cc_major, int cuda_cc_minor)
{
    std::vector<int>& case_permutations = *p_case_permutations;
    case_permutations.clear();
    std::vector<KernelCaseEntry<KernelCase>> case_entries;

    for (int case_i = 0; case_i < KernelCase::num_user_cases + KernelCase::num_builtin_cases; ++case_i) {
        KernelCaseEntry<KernelCase> entry{};
        if (case_i < KernelCase::num_user_cases) {
            entry.p_case = &KernelCase::user_cases[case_i];
            entry.is_builtin = false;
        }
        else {
            entry.p_case = &KernelCase::builtin_cases[case_i - KernelCase::num_user_cases];
            entry.is_builtin = true;
        }

        if (false) {
            // TODO: add mechanism to filter out kernels.
        }
        else if (!cuda_arch_supports(entry.p_case->cuda_arch, cuda_cc_major, cuda_cc_minor)) {
            // Skip unsupported architectures.
            if (false) {
                printf("    Skipped: %s\n", entry.p_case->proc_name);
            }
        }
        else {
            // Schedule this kernel for testing.
            for (int K_split = 1; K_split <= 16; K_split *= 2) {
                if (K_split <= entry.p_case->K_split_max && K_split % entry.p_case->K_split_divisor == 0) {
                    entry.K_split = K_split;
                    case_entries.push_back(entry);
                    case_permutations.push_back(int(case_permutations.size()));
                }
            }
        }
    }

    return case_entries;
}

TestDataConfig get_data_config(int trial_i, int K)
{
    TestDataCode A_code = TestDataCode::random;
    TestDataCode B_code = TestDataCode::random;
    TestCheckMode check_mode = TestCheckMode::none;

    // We will do testing on up to first 4 warmup iterations only.
    // Warmups may use different test data; timed iterations always use random data.
    if (trial_i < num_warmup) {
        switch (trial_i) {
          case 0:
            A_code = TestDataCode::signs_only;
            B_code = TestDataCode::signs_only;
            check_mode = K <= 8192 ? TestCheckMode::exact : TestCheckMode::approximate;
            break;
          case 1:
            A_code = TestDataCode::tiled_numbers;
            B_code = TestDataCode::batch_index_identity;
            check_mode = TestCheckMode::approximate;
            break;
          case 2:
            A_code = TestDataCode::batch_index_identity;
            B_code = TestDataCode::tiled_numbers;
            check_mode = TestCheckMode::approximate;
            break;
          case 3:
            check_mode = TestCheckMode::approximate;
            break;
        }
    }

    return TestDataConfig{A_code, B_code, check_mode};
}

template <typename KernelCase>
void summarize_entry(const MainData& main_data, KernelCaseEntry<KernelCase>& entry, bool is_first)
{
    fprintf(main_data.json_file, "     %c{\n", is_first ? ' ' : ',');
    fprintf(main_data.json_file, "        \"proc\": \"%s\",\n", entry.p_case->proc_name);
    fprintf(main_data.json_file, "        \"K_split\": %i,\n", entry.K_split);
    fprintf(main_data.json_file, "        \"is_builtin\": %s,\n", entry.is_builtin ? "true" : "false");
    fprintf(main_data.json_file, "        \"flops_samples\": [");
    bool need_comma = false;
    for (double flops : entry.flops_samples) {
        fprintf(main_data.json_file, "%s%.12e", need_comma ? ", " : "", flops);
        need_comma = true;
    }
    fprintf(main_data.json_file, "],\n");

    // Average the IQR.
    std::sort(entry.flops_samples.begin(), entry.flops_samples.end());
    size_t iqr_begin = entry.flops_samples.size() / 4u;
    size_t iqr_end = entry.flops_samples.size() * 3u / 4u;
    if (iqr_begin >= iqr_end) {
        iqr_begin = 0;
        iqr_end = entry.flops_samples.size();
    }
    double accum = 0;
    for (size_t i = iqr_begin; i < iqr_end; ++i) {
        accum += entry.flops_samples[i];
    }
    const int num_samples = int(iqr_end) - int(iqr_begin);
    const double flops_iqr = num_samples != 0 ? accum / (iqr_end - iqr_begin) : 0.0;

    // Write flops_iqr to JSON; last item doesn't have a comma.
    fprintf(main_data.json_file, "        \"flops_iqr\": %.12e\n", flops_iqr);

    // Print info to stdout as well.
    const int color_code = entry.is_builtin ? 32 : entry.K_split > 1 ? 36 : 0;
    printf("%8.3f \x1b[%imTFLOPS\x1b[0m; K/%i, %s (samples=%i)\n",
            flops_iqr / 1e12, color_code, entry.K_split, entry.p_case->proc_name, num_samples);
    fprintf(main_data.json_file, "     }\n");
}


void generate_gemm_plot_samples(const MainData& main_data, const GemmPlotInput& plot_input)
{
    bool need_comma = false;
    std::vector<int> case_permutations;
    std::vector<KernelCaseEntry<GemmCase>> gemm_case_entries = generate_cases<GemmCase>(
            &case_permutations, main_data.cuda_cc_major, main_data.cuda_cc_minor);

    for (GemmPlotSize plot_size : plot_input.sizes) {
        const int L = plot_size.L;
        const int M = plot_size.M;
        const int N = plot_size.N;
        const int K = plot_size.K;

        warn_if_no_json();
        fprintf(main_data.json_file, "    %c{\"L\": %i, \"M\": %i, \"N\": %i, \"K\": %i, \"kernels\": [\n",
                need_comma ? ',' : ' ', L, M, N, K);
        need_comma = true;

        AsyncDeleter deleter{main_data.stream};
        int union_flags = 0;
        int union_not_flags = 0;
        for (const KernelCaseEntry<GemmCase>& entry : gemm_case_entries) {
            const int flags = entry.p_case->flags;
            union_flags |= flags;
            union_not_flags |= ~flags;
        }

        printf("\n\x1b[34m\x1b[1mGEMM:\x1b[0m\n");
        printf("L = %i, MNK = [%i, %i, %i]\n", L, M, N, K);
        std::unique_ptr<float[], AsyncDeleter> unique_L2_shred_memory = alloc_f32(1, 1, L2_shred_bytes / 4u, deleter);
        std::unique_ptr<float[], AsyncDeleter> unique_C_test = alloc_f32(L, M, N, deleter);
        std::unique_ptr<float[], AsyncDeleter> unique_C_expected = alloc_f32(L, M, N, deleter);
        std::unique_ptr<float[], AsyncDeleter> unique_A_row_major;
        std::unique_ptr<float[], AsyncDeleter> unique_B_row_major;
        std::unique_ptr<float[], AsyncDeleter> unique_A_col_major;
        std::unique_ptr<float[], AsyncDeleter> unique_B_col_major;

        // Allocate A column major, or B row major, only if some test case requires it.
        // A row major, B column major is required by our usage of cublas to generate expected output.
        if (true) {
            unique_A_row_major = alloc_f32(L, M, K, deleter);
        }
        if ((union_not_flags & A_row_major_flag)) {
            unique_A_col_major = alloc_f32(L, M, K, deleter);
        }
        if ((union_flags & B_row_major_flag)) {
            unique_B_row_major = alloc_f32(L, N, K, deleter);
        }
        if (true) {
            unique_B_col_major = alloc_f32(L, N, K, deleter);
        }

        GemmTestResources resources{};
        resources.cublasH = main_data.cublasH;
        resources.start_event = main_data.start_event;
        resources.end_event = main_data.end_event;
        resources.A_row_major = unique_A_row_major.get();
        resources.A_col_major = unique_A_col_major.get();
        resources.B_row_major = unique_B_row_major.get();
        resources.B_col_major = unique_B_col_major.get();
        resources.C_test = unique_C_test.get();
        resources.C_expected = unique_C_expected.get();
        resources.L2_shred_bytes = L2_shred_bytes;
        resources.L2_shred_memory = unique_L2_shred_memory.get();

        for (int trial_i = 0; trial_i < num_warmup + num_timed; ++trial_i) {
            TestDataConfig data_config = get_data_config(trial_i, K);

            // Initialize test data on every warmup iteration, and the first timed iteration.
            // Additional test data generation is not needed as timed iterations always use the same data.
            if (trial_i < num_warmup + 1) {
                GemmSize size{};
                size.L = L;
                size.M = M;
                size.N = N;
                size.K_split = 1;
                size.K_cluster = K;
                init_test_data(resources, size, data_config.A_code, data_config.B_code);
            }

            // Test kernels in a random order.
            shuffle_ints(case_permutations, trial_i + 27182818, M * N);
            for (auto entry_index : case_permutations) {
                KernelCaseEntry<GemmCase>& entry = gemm_case_entries.at(entry_index);
                const GemmCase& gemm_case = *entry.p_case;
                const int K_split = entry.K_split;
                GemmSize size{};
                size.L = L;
                size.M = M;
                size.N = N;
                size.K_split = K_split;
                size.K_cluster = K / K_split;
                if (size.K_split * size.K_cluster != K) {
                    // We only support exact divisibilty for K_split for now.
                    continue;
                }
                if (gemm_case.supports(size)) {
                    TestResult result = run_gemm_case(gemm_case, resources, size, data_config.check_mode);
                    global_all_passed &= result.passed;
                    if (trial_i >= num_warmup) {
                        entry.flops_samples.push_back(result.flops);
                    }
                    if (false) {
                        printf("TFLOPS=%g, L=%i, MNK=[%i, %i, %i], K/%i, %s\n",
                                result.flops / 1e12, L, M, N, K, K_split, gemm_case.proc_name);
                    }
                }
            };
            fprintf(stderr, ".");
        }
        fprintf(stderr, "\n");

        for (KernelCaseEntry<GemmCase>& entry: gemm_case_entries) {
            summarize_entry(main_data, entry, &entry == &gemm_case_entries[0]);

            // Clear flops samples vector for the next problem size.
            entry.flops_samples.clear();
        }

        fprintf(main_data.json_file, "    ]}\n");
    }
}


void generate_gemv_plot_samples(const MainData& main_data, const GemvPlotInput& plot_input)
{
    bool need_comma = false;
    std::vector<int> case_permutations;
    std::vector<KernelCaseEntry<GemvCase>> gemv_case_entries = generate_cases<GemvCase>(
            &case_permutations, main_data.cuda_cc_major, main_data.cuda_cc_minor);

    for (GemvPlotSize plot_size : plot_input.sizes) {
        const int L = 1;
        const int M = plot_size.M;
        const int K = plot_size.K;

        warn_if_no_json();
        fprintf(main_data.json_file, "    %c{\"L\": %i, \"M\": %i, \"K\": %i, \"kernels\": [\n",
                need_comma ? ',' : ' ', L, M, K);
        need_comma = true;

        AsyncDeleter deleter{main_data.stream};
        printf("\n\x1b[34m\x1b[1mGEMV:\x1b[0m\n");
        printf("MK = [%i, %i]\n", M, K);

        std::unique_ptr<float[], AsyncDeleter> unique_L2_shred_memory = alloc_f32(1, 1, L2_shred_bytes / 4u, deleter);
        std::unique_ptr<float[], AsyncDeleter> unique_A = alloc_f32(L, M, K, deleter);
        std::unique_ptr<float[], AsyncDeleter> unique_x = alloc_f32(L, 1, K, deleter);
        std::unique_ptr<float[], AsyncDeleter> unique_y_test = alloc_f32(L, M, 1, deleter);
        std::unique_ptr<float[], AsyncDeleter> unique_y_expected = alloc_f32(L, M, 1, deleter);

        GemvTestResources resources{};
        resources.cublasH = main_data.cublasH;
        resources.start_event = main_data.start_event;
        resources.end_event = main_data.end_event;
        resources.A = unique_A.get();
        resources.x = unique_x.get();
        resources.y_test = unique_y_test.get();
        resources.y_expected = unique_y_expected.get();
        resources.L2_shred_bytes = L2_shred_bytes;
        resources.L2_shred_memory = unique_L2_shred_memory.get();

        for (int trial_i = 0; trial_i < num_warmup + num_timed; ++trial_i) {
            TestDataConfig data_config = get_data_config(trial_i, K);

            // Initialize test data on every warmup iteration, and the first timed iteration.
            // Additional test data generation is not needed as timed iterations always use the same data.
            if (trial_i < num_warmup + 1) {
                GemvSize size{};
                size.M = M;
                size.K = K;
                init_test_data(resources, size, data_config.A_code, data_config.B_code);
            }

            // Test kernels in a random order.
            shuffle_ints(case_permutations, trial_i + 27182818, M * K);
            for (auto entry_index : case_permutations) {
                KernelCaseEntry<GemvCase>& entry = gemv_case_entries.at(entry_index);
                const GemvCase& gemv_case = *entry.p_case;
                GemvSize size{};
                size.M = M;
                size.K = K;
                if (gemv_case.supports(size)) {
                    TestResult result = run_gemv_case(gemv_case, resources, size, data_config.check_mode);
                    global_all_passed &= result.passed;
                    if (trial_i >= num_warmup) {
                        entry.flops_samples.push_back(result.flops);
                    }
                }
            };
            fprintf(stderr, ".");
        }
        fprintf(stderr, "\n");

        for (KernelCaseEntry<GemvCase>& entry: gemv_case_entries) {
            summarize_entry(main_data, entry, &entry == &gemv_case_entries[0]);

            // Clear flops samples vector for the next problem size.
            entry.flops_samples.clear();
        }

        fprintf(main_data.json_file, "    ]}\n");
    }
}

int Main(int argc, char** argv)
{
    MainData main_data{};
    const char* json_filename = "/dev/null";
    if (argc == 2) {
        json_filename = argv[1];
    }
    else if (argc > 2) {
        fprintf(stderr, "%s: accepts on argument, JSON output filename\n", argv[0]);
        return 1;
    }
    else {
        warn_if_no_json = []
        {
            fprintf(stderr, "\x1b[31m\x1b[1mNOTE:\x1b[0m no JSON output given; sending to /dev/null\n");
        };
    }
    main_data.json_file = fopen(json_filename, "wb");
    if (!main_data.json_file) {
        fprintf(stderr, "%s: %s, %s\n", argv[0], strerror(errno), json_filename);
        return 1;
    }

    cudaSetDevice(0);
    cudaDeviceGetAttribute(&main_data.cuda_cc_major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&main_data.cuda_cc_minor, cudaDevAttrComputeCapabilityMinor, 0);
    const bool is_h100 = main_data.cuda_cc_major == 9 && main_data.cuda_cc_minor == 0;
    fprintf(stderr, "is_h100: %i\n", is_h100);

    if (const cudaError_t err = cudaEventCreate(&main_data.start_event)) {
        throw std::runtime_error("cudaEventCreate failed\n");
    }
    if (const cudaError_t err = cudaEventCreate(&main_data.end_event)) {
        throw std::runtime_error("cudaEventCreate failed\n");
    }
    main_data.stream = cudaStream_t{};  // Not passed to test cases currently.
    CUBLAS_CHECK(cublasCreate(&main_data.cublasH));
    CUBLAS_CHECK(cublasSetStream(main_data.cublasH, main_data.stream));
    CUBLAS_CHECK(cublasSetMathMode(main_data.cublasH, CUBLAS_TF32_TENSOR_OP_MATH));

    fprintf(main_data.json_file, "[\n");
    bool need_comma = false;
    auto begin_json_plot_object = [&] (const auto& plot_input)
    {
        fprintf(
                main_data.json_file,
                " %c{\"name\": \"%s\", \"title\": \"%s %s\", \"x_axis\": \"%s\", \"samples\": [\n",
                need_comma ? ',' : ' ',
                plot_input.name.c_str(),
                is_h100 ? "sm_90a" : "sm_80",
                plot_input.title.c_str(),
                plot_input.x_axis.c_str()
        );
        need_comma = true;
    };
    auto end_json_plot_object = [&]
    {
        fprintf(main_data.json_file, "  ]}\n");
    };

    if (GemmCase::num_user_cases > 0) {
        for (const GemmPlotInput& plot_input : generate_gemm_plot_inputs(is_h100)) {
            begin_json_plot_object(plot_input);
            generate_gemm_plot_samples(main_data, plot_input);
            end_json_plot_object();
        }
    }
    else {
        fprintf(stderr, "No user GemmCase instances, skipping...\n");
    }

    if (GemvCase::num_user_cases > 0) {
        for (const GemvPlotInput& plot_input : generate_gemv_plot_inputs()) {
            begin_json_plot_object(plot_input);
            generate_gemv_plot_samples(main_data, plot_input);
            end_json_plot_object();
        }
    }
    else {
        fprintf(stderr, "No user GemvCase instances, skipping...\n");
    }

    fprintf(main_data.json_file, "]\n");

    if (global_all_passed) {
        printf("All tests passed.\n");
    }
    else {
        printf("\x1b[31m\x1b[1mFAILED:\x1b[0m Not all test cases passed!\n");
    }

    fclose(main_data.json_file);
    cublasDestroy(main_data.cublasH);
    return global_all_passed ? 0 : 1;
}

}  // end namespace

int main(int argc, char** argv)
{
    return sporkbench::Main(argc, argv);
}
