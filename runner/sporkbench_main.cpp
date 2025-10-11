#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <stdexcept>
#include <stdio.h>
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

struct TestDataConfig
{
    TestDataCode A_code;
    TestDataCode B_code;
    TestCheckMode check_mode;
};

// (L, M, N, K)
static std::vector<std::tuple<int, int, int, int>> generate_gemm_sizes(bool is_h100)
{
    if (is_h100) {
        return {
            {1, 2048, 2048, 2048}, {4, 2048, 2048, 2048},
            {1, 4096, 4096, 4096}, {4, 4096, 4096, 4096},
            {1, 7680, 7680, 8192}, {4, 7680, 7680, 8192},
            {1, 2816, 768, 65536}, {4, 2816, 768, 65536},
        };
    }
    else {
        return {
            {1, 1536, 1536, 1536},
            {1, 3840, 1536, 4096},
            {1, 1536, 3840, 4096},
        };
    }
}

static std::vector<std::tuple<int, int>> generate_gemv_sizes(bool is_h100)
{
    (void)is_h100;

    return {
        {1024, 2048},
        {2048, 2048},
        {4096, 2048},
        {4096, 4096},
    };
}

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
void summarize_entry(KernelCaseEntry<KernelCase>& entry)
{
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
    const double flops = num_samples != 0 ? accum / (iqr_end - iqr_begin) : 0.0;

    // Print info.
    const int color_code = entry.is_builtin ? 32 : entry.K_split > 1 ? 36 : 0;
    printf("%8.3f \x1b[%imTFLOPS\x1b[0m; K/%i, %s (samples=%i)\n",
            flops / 1e12, color_code, entry.K_split, entry.p_case->proc_name, num_samples);
}


int Main(int argc, char** argv)
{
    cudaSetDevice(0);

    int cuda_cc_major{}, cuda_cc_minor{};
    cudaDeviceGetAttribute(&cuda_cc_major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&cuda_cc_minor, cudaDevAttrComputeCapabilityMinor, 0);
    const bool is_h100 = cuda_cc_major == 9 && cuda_cc_minor == 0;
    fprintf(stderr, "is_h100: %i\n", is_h100);

    std::vector<int> case_permutations;
    cudaEvent_t start_event, end_event;
    if (const cudaError_t err = cudaEventCreate(&start_event)) {
        throw std::runtime_error("cudaEventCreate failed\n");
    }
    if (const cudaError_t err = cudaEventCreate(&end_event)) {
        throw std::runtime_error("cudaEventCreate failed\n");
    }
    cudaStream_t stream{};  // Not passed to test cases currently.
    cublasHandle_t cublasH{};
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_TF32_TENSOR_OP_MATH));

    std::vector<KernelCaseEntry<GemmCase>> gemm_case_entries = generate_cases<GemmCase>(
            &case_permutations, cuda_cc_major, cuda_cc_minor);

    for (std::tuple<int, int, int, int> lmnk: generate_gemm_sizes(is_h100)) {
        auto [L, M, N, K] = lmnk;
        AsyncDeleter deleter{stream};
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
        resources.cublasH = cublasH;
        resources.start_event = start_event;
        resources.end_event = end_event;
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
            summarize_entry(entry);

            // Clear flops samples vector for the next problem size.
            entry.flops_samples.clear();
        }
    }

    std::vector<KernelCaseEntry<GemvCase>> gemv_case_entries = generate_cases<GemvCase>(
            &case_permutations, cuda_cc_major, cuda_cc_minor);

    for (std::tuple<int, int> mk: generate_gemv_sizes(is_h100)) {
        const int L = 1;
        auto [M, K] = mk;
        AsyncDeleter deleter{stream};
        printf("\n\x1b[34m\x1b[1mGEMV:\x1b[0m\n");
        printf("MK = [%i, %i]\n", M, K);

        std::unique_ptr<float[], AsyncDeleter> unique_L2_shred_memory = alloc_f32(1, 1, L2_shred_bytes / 4u, deleter);
        std::unique_ptr<float[], AsyncDeleter> unique_A = alloc_f32(L, M, K, deleter);
        std::unique_ptr<float[], AsyncDeleter> unique_x = alloc_f32(L, 1, K, deleter);
        std::unique_ptr<float[], AsyncDeleter> unique_y_test = alloc_f32(L, M, 1, deleter);
        std::unique_ptr<float[], AsyncDeleter> unique_y_expected = alloc_f32(L, M, 1, deleter);

        GemvTestResources resources{};
        resources.cublasH = cublasH;
        resources.start_event = start_event;
        resources.end_event = end_event;
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
            summarize_entry(entry);

            // Clear flops samples vector for the next problem size.
            entry.flops_samples.clear();
        }
    }

    if (global_all_passed) {
        printf("All tests passed.\n");
    }
    else {
        printf("\x1b[31m\x1b[1mFAILED:\x1b[0m Not all test cases passed!\n");
    }

    cublasDestroy(cublasH);
    return global_all_passed ? 0 : 1;
}

}  // end namespace

int main(int argc, char** argv)
{
    return sporkbench::Main(argc, argv);
}
