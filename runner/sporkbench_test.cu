#include "sporkbench_test.hpp"

#include <bit>
#include <cassert>
#include <cublas_v2.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>
#include <vector>

#include "sporkbench_cublas_gemm.hpp"
#include "sporkbench_pcg3d.hpp"

#include "sporkbench_kittens_mha_Sm90a.hpp"  // TODO replace with cuDNN

namespace sporkbench {

const TestResult TestResult::passed_0_flops{true, 0};

namespace sporkbench_test {

// k_major means that K is the "fast" dimension (i.e. K stride is 1, MN stride is K).
// !k_major means that K is the "slow" dimension (i.e. K stride is MN, MN stride is 1).
// This is Nvidia's term for it, and I use it consistently, even though I think it's a poor choice.
template <typename T>
__global__ void device_init_test_data(
        T* d_tensor, uint32_t batch_size, uint32_t MN, uint32_t K, bool k_major, TestDataCode code)
{
    uint32_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t tid_z = threadIdx.z + blockIdx.z * blockDim.z;
    const uint32_t major_extent = k_major ? MN : K;  // k_major is a stooopid name.
    const uint32_t minor_extent = k_major ? K : MN;
    for (uint32_t z = tid_z; z < batch_size; z += blockDim.z * gridDim.z) {
        for (uint32_t y = tid_y; y < major_extent; y += blockDim.y * gridDim.y) {
            for (uint32_t x = tid_x; x < minor_extent; x += blockDim.x * gridDim.x) {
                const auto k = k_major ? x : y;
                const auto mn = k_major ? y : x;
                T value;
                switch (code) {
                  case TestDataCode::batch_index_identity:
                    value = k == mn ? T(1 + z) : T(0);
                    break;
                  case TestDataCode::tiled_numbers:
                    if constexpr (sizeof(T) >= 2) {
                        value = T((k % 64) + 100 * (mn % 64));
                    }
                    else {
                        value = T((k % 4) + 4 * (mn % 4));
                    }
                    break;
                  case TestDataCode::signs_only:
                    {
                        // uniform choice between -1, 0, 1.
                        value = T(int(pcg3d(k, mn, z + 20010106) % 3) - 1);
                    }
                    break;
                  case TestDataCode::random_no_outliers:
                  case TestDataCode::random_with_outliers:
                    {
                        const auto randbits = pcg3d(k, mn, z + 20010106);
                        if (code == TestDataCode::random_with_outliers && randbits % 100'000u == 0) {
                            // 1 in 100'000 chance of a "big" value.
                            // This greatly reduces the chance that a genuine bug is mistaken for fp error.
                            value = sizeof(T) >= 4 ? T(1000) : T(24);
                        }
                        else if (randbits % 4u != 0u) {
                            value = T(0);  // 75% chance of a 0
                        }
                        else {
                            // 25% chance of random value [0, 1], biased towards small numbers.
                            value = T((pcg3d(k, mn, 19980724) % 1'000'000) * 1e-6f);
                            if constexpr(sizeof(T) >= 2) {
                                value = (value * value) * (value * value);
                            }
                        }
                    }
                    break;
                }
                d_tensor[z * major_extent * minor_extent + y * minor_extent + x] = value;
            }
        }
    }
}


template <typename T>
void launch_init_test_data(
        T* d_tensor, uint32_t batch_size, uint32_t MN, uint32_t K, bool k_major,
        TestDataCode code, cudaStream_t stream)
{
    const uint32_t major_extent = k_major ? MN : K;  // k_major is a stooopid name.
    const uint32_t minor_extent = k_major ? K : MN;
    dim3 grid{(minor_extent + 63u) / 64u, (major_extent + 3u) / 4u, batch_size};
    dim3 block{64, 4, 1};
    device_init_test_data<<<grid, block, 0, stream>>>(d_tensor, batch_size, MN, K, k_major, code);
}


__global__ void device_compare_tensor_test_init_bitfield(unsigned long long* d_bitfield)
{
    *d_bitfield = UINT64_MAX;
}

struct TestTensorSize
{
    int batches;
    int heads;
    int rows;
    int cols;

    __host__ __device__ size_t row_major_index(int batch, int head, int row, int col) const
    {
        return ((size_t(batch) * heads + head) * rows + row) * cols + col;
    }

    __host__ __device__ size_t col_major_index(int batch, int head, int row, int col) const
    {
        return ((size_t(batch) * heads + head) * cols + col) * rows + row;
    }

    __host__ __device__ void unpack_row_major(size_t linear_index, int* batch, int* head, int* row, int* col) const
    {
        *col = int(linear_index % size_t(cols));
        linear_index /= size_t(cols);
        *row = int(linear_index % size_t(rows));
        linear_index /= size_t(rows);
        *head = int(linear_index % size_t(heads));
        linear_index /= size_t(heads);
        *batch = int(linear_index);
    }

    __host__ __device__ void unpack_col_major(size_t linear_index, int* batch, int* head, int* row, int* col) const
    {
        *row = int(linear_index % size_t(rows));
        linear_index /= size_t(rows);
        *col = int(linear_index % size_t(cols));
        linear_index /= size_t(cols);
        *head = int(linear_index % size_t(heads));
        linear_index /= size_t(heads);
        *batch = int(linear_index);
    }
};

struct AttnStatsVectorSize
{
    int Batch, KV_Heads, Groups, SeqLen, Hdim;
};

TestTensorSize to_test_tensor_size(GemmSize size)
{
    return TestTensorSize{size.L, 1, size.M, size.N};
}

TestTensorSize to_test_tensor_size(GemvSize size)
{
    return TestTensorSize{1, 1, 1, size.M};
}

TestTensorSize to_test_tensor_size(AttnFwdSize size)
{
    const int qo_heads = size.KV_Heads * size.Groups;
    return TestTensorSize{size.Batch, qo_heads, size.SeqLen, size.Hdim};
}

TestTensorSize to_test_tensor_size(AttnStatsVectorSize size)
{
    const int qo_heads = size.KV_Heads * size.Groups;
    return TestTensorSize{size.Batch, qo_heads, size.SeqLen, 1};
}

void print_problem_size(GemmSize size)
{
    printf("L=%i, MNK=[%i, %i, %i], K_split=%i", size.L, size.M, size.N, size.K_cluster * size.K_split, size.K_split);
}

void print_problem_size(GemvSize size)
{
    printf("M=%i, K=%i", size.M, size.K);
}

void print_problem_size(AttnFwdSize size)
{
    printf(
        "Batch=%i, KV_Heads=%i, Groups=%i, SeqLen=%i, Hdim=%i\n",
        size.Batch, size.KV_Heads, size.Groups, size.SeqLen, size.Hdim
    );
}

void print_problem_size(AttnStatsVectorSize size)
{
    printf(
        "Batch=%i, KV_Heads=%i, Groups=%i, SeqLen=%i, Hdim=%i\n",
        size.Batch, size.KV_Heads, size.Groups, size.SeqLen, size.Hdim
    );
}

// Requires that *d_bitfield is initialized to UINT64_MAX.
// Compare the two equal-sized matrices and, if any comparison failures, put the coordinates of the wrong value
// into *d_bitfield, packed as its linear_index into the d_expected array.
// d_expected and d_test are described by the row_major flag.
//
// NOTE: d_expected used to be hard-wired as column major. If any comments describe this, they are outdated.
template <typename Test, typename Expected>
__global__ void device_compare_tensor_test(TestTensorSize size, const Test* d_test, const Expected* d_expected,
                                           bool row_major, bool exact, unsigned long long* d_bitfield)
{
    uint32_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t tid_z = threadIdx.z + blockIdx.z * blockDim.z;
    const uint32_t batches_heads = uint32_t(size.batches) * uint32_t(size.heads);
    for (uint32_t b_h = tid_z; b_h < batches_heads; b_h += blockDim.z * gridDim.z) {
        uint32_t batch = b_h / uint32_t(size.heads);
        uint32_t head = b_h % uint32_t(size.heads);
        for (uint32_t m = tid_y; m < uint32_t(size.rows); m += blockDim.y * gridDim.y) {
            for (uint32_t n = tid_x; n < uint32_t(size.cols); n += blockDim.x * gridDim.x) {
                size_t linear_index;
                if (row_major) {
                    linear_index = size.row_major_index(batch, head, m, n);
                }
                else {
                    linear_index = size.col_major_index(batch, head, m, n);
                }
                bool correct = false;
                if (exact) {
                    correct = d_test[linear_index] == d_expected[linear_index];
                }
                else {
                    float f_test = float(d_test[linear_index]);
                    float f_expected = float(d_expected[linear_index]);
                    if (fabsf(f_test - f_expected) <= 1.0f / 1024) {
                        // Any absolute difference <= 1/1024 is a pass.
                        // Note the <= will always be false for NaN.
                        correct = true;
                    }
                    else if (f_test * f_expected >= 0.0f) {
                        // Allow same-sign values with small relative ratio.
                        f_test = fabsf(f_test);
                        f_expected = fabsf(f_expected);
                        const float Min = fminf(f_test, f_expected);
                        const float Max = fmaxf(f_test, f_expected);
                        const float eps = sizeof(Test) >= 4 ? 1.0f / 128 : 12.0f / 128;
                        correct = Max == 0 || Max / Min <= (1.0f + eps);
                    }
                    else {
                        correct = false;
                    }
                }
                if (!correct) {
                    atomicMin(d_bitfield, (unsigned long long)linear_index);
                }
            }
        }
    }
}

template <typename Test, typename Expected>
__device__ void print_tensor_neighborhood(TestTensorSize size, const Test* d_test, const Expected* d_expected,
                                          bool row_major, uint32_t batch, uint32_t head, uint32_t m, uint32_t n)
{
    uint32_t m_min = m < 2 ? 0u : m - 2;
    uint32_t m_max = m + 2 >= size.rows ? size.rows - 1u : m + 2;
    uint32_t n_min = n < 2 ? 0u : n - 2;
    uint32_t n_max = n + 2 >= size.cols ? size.cols - 1u : n + 2;

    for (uint32_t cm = m_min; cm <= m_max; cm++) {
        for (uint32_t cn = n_min; cn <= n_max; cn++) {
            if (cn == n && cm == m) {
                printf("\x1b[1m");
            }
            size_t linear_index;
            if (row_major) {
                linear_index = size.row_major_index(batch, head, cm, cn);
            }
            else {
                linear_index = size.col_major_index(batch, head, cm, cn);
            }
            const double f_test = static_cast<double>(d_test[linear_index]);
            const double f_expected = static_cast<double>(d_expected[linear_index]);
            printf("[%6g, %6g]  ", f_test, f_expected);
            if (cn == n && cm == m) {
                printf("\x1b[0m");
            }
        }
        printf("\n");
    }
}



// Print info on wrong value from function above.
template <typename Test, typename Expected>
__global__ void device_compare_tensor_test_print(TestTensorSize size, const Test* d_test, const Expected* d_expected,
                                                 bool row_major, unsigned long long* d_bitfield)
{
    unsigned long long linear_index = *d_bitfield;
    if (linear_index != UINT64_MAX) {
        int batch, head, row, col;
        if (row_major) {
            size.unpack_row_major(*d_bitfield, &batch, &head, &row, &col);
        }
        else {
            size.unpack_col_major(*d_bitfield, &batch, &head, &row, &col);
        }
        const double f_test = static_cast<double>(d_test[linear_index]);
        const double f_expected = static_cast<double>(d_expected[linear_index]);
        if (size.heads == 1) {
            printf(
                "\x1b[1m[batch=%i, m=%i, n=%i]\x1b[0m %g != %g (test != expected)\n",
                batch, row, col, f_test, f_expected);
        }
        else {
            printf(
                "\x1b[1m[batch=%i, head=%i, row=%i, col=%i]\x1b[0m %g != %g (test != expected)\n",
                batch, head, row, col, f_test, f_expected);
        }

        print_tensor_neighborhood(size, d_test, d_expected, row_major, batch, head, row, col);
    }
}

template <typename ProblemSize, typename Test, typename Expected>
bool launch_device_compare_tensor(
        ProblemSize problem_size, const char* proc_name, const Test* d_test, const Expected* d_expected,
        bool row_major, bool exact, cudaStream_t stream)
{
    const auto test_size = to_test_tensor_size(problem_size);
    static_assert(std::is_same_v<decltype(test_size), const TestTensorSize>);
    dim3 grid(
        unsigned(test_size.cols + 63u) / 64u,
        unsigned(test_size.rows + 3u) / 4u,
        unsigned(test_size.batches * test_size.heads)
    );
    dim3 block(64, 4, 1);
    unsigned long long* d_bitfield = 0;
    cudaMallocAsync(&d_bitfield, sizeof(*d_bitfield), stream);
    if (!d_bitfield) {
        fprintf(stderr, "Alloc of d_bitfield failed: %s:%i\n", __FILE__, __LINE__);
        return false;
    }
    device_compare_tensor_test_init_bitfield<<<1, 1, 0, stream>>>(d_bitfield);
    device_compare_tensor_test<<<grid, block, 0, stream>>>(test_size, d_test, d_expected, row_major, exact, d_bitfield);
    unsigned long long h_bitfield;
    cudaMemcpy(&h_bitfield, d_bitfield, sizeof(h_bitfield), cudaMemcpyDeviceToHost);
    if (h_bitfield != UINT64_MAX) {
        printf("\x1b[31m\x1b[1mFAILED:\x1b[0m %s ", proc_name);
        print_problem_size(problem_size);
        printf("\n");
        fflush(stdout);
        device_compare_tensor_test_print<<<1, 1, 0, stream>>>(test_size, d_test, d_expected, row_major, d_bitfield);
        cudaStreamSynchronize(stream);  // flush stdout.
    }
    cudaFreeAsync(d_bitfield, stream);
    return h_bitfield == UINT64_MAX;
}


template <typename Ctype, typename ABtype>
double run_gemm_case_impl(
        const GemmCaseT<Ctype, ABtype>& gemm_case,
        const GemmTestResourcesT<Ctype, ABtype>& resources,
        GemmSize size,
        cudaStream_t stream)
{
    // Update readme if you change the testing methodology.
    const uint32_t L = uint32_t(size.L);
    const uint32_t M = uint32_t(size.M);
    const uint32_t N = uint32_t(size.N);
    const uint32_t K = uint32_t(size.K_split * size.K_cluster);
    const ABtype* A = (gemm_case.flags & A_row_major_flag) ? resources.A_row_major : resources.A_col_major;
    assert(A);
    const ABtype* B = (gemm_case.flags & B_row_major_flag) ? resources.B_row_major : resources.B_col_major;
    assert(B);

    cudaMemsetAsync(resources.L2_shred_memory, 0xCC, resources.L2_shred_bytes, stream);
    cudaEventRecord(resources.start_event, stream);
    assert(stream == 0);  // Change run_function to take stream argument.
    gemm_case.run_function(resources.cublasH, size, A, B, resources.C_test);
    cudaEventRecord(resources.end_event, stream);

    cudaStreamSynchronize(stream);
    float ms;
    cudaEventElapsedTime(&ms, resources.start_event, resources.end_event);

    const double flops = double(L) * M * N * K * 2000.0 / ms;
    return flops;
}

double run_gemv_case_impl(
        const GemvCase& gemv_case, const GemvTestResources& resources, GemvSize size, cudaStream_t stream)
{
    // Update readme if you change the testing methodology.
    cudaMemsetAsync(resources.L2_shred_memory, 0xCC, resources.L2_shred_bytes, stream);
    cudaEventRecord(resources.start_event, stream);
    assert(stream == 0);  // Change run_function to take stream argument.
    gemv_case.run_function(resources.cublasH, size, resources.A, resources.x, resources.y_test);
    cudaEventRecord(resources.end_event, stream);

    cudaStreamSynchronize(stream);
    float ms;
    cudaEventElapsedTime(&ms, resources.start_event, resources.end_event);

    const double flops = double(size.M) * size.K * 2000.0 / ms;
    return flops;
}

template <typename T_type, typename L_type, int Hdim, bool Causal>
double run_attn_fwd_case_impl(
        const AttnFwdCaseT<T_type, L_type, Hdim, Causal>& attn_case,
        const AttnFwdTestResourcesT<T_type, L_type, Hdim, Causal>& resources,
        AttnFwdSize size,
        cudaStream_t stream)
{
    // Update readme if you change the testing methodology.
    cudaMemsetAsync(resources.L2_shred_memory, 0xCC, resources.L2_shred_bytes, stream);
    cudaEventRecord(resources.start_event, stream);
    assert(stream == 0);  // Change run_function to take stream argument.
    attn_case.run_function(
            size, resources.d_O_test, resources.d_l_vec_test,
            resources.d_Q, resources.d_K, resources.d_V);
    cudaEventRecord(resources.end_event, stream);

    cudaStreamSynchronize(stream);
    float ms;
    cudaEventElapsedTime(&ms, resources.start_event, resources.end_event);

    // Only counted tensor flops
    const auto qk_macs = double(size.SeqLen) * size.SeqLen * Hdim;
    const auto so_macs = double(size.SeqLen) * size.SeqLen * Hdim;
    const double flops = double(size.Batch * size.KV_Heads * size.Groups) * (qk_macs + so_macs) * 2000.0 / ms;
    return flops;
}

}  // end namespace sporkbench_test

template <typename Ctype, typename ABtype>
void init_test_data_impl(
        const GemmTestResourcesT<Ctype, ABtype>& resources,
        GemmSize size,
        TestDataCode A_code,
        TestDataCode B_code)
{
    using namespace ::sporkbench::sporkbench_test;
    const cudaStream_t stream = 0;
    const auto K = size.K_cluster * size.K_split;

    if (resources.A_row_major) {
        launch_init_test_data(resources.A_row_major, size.L, size.M, K, true, A_code, stream);
    }
    if (resources.A_col_major) {
        launch_init_test_data(resources.A_col_major, size.L, size.M, K, false, A_code, stream);
    }
    if (resources.B_row_major) {
        // false is the K-major flag. Not "row major".
        launch_init_test_data(resources.B_row_major, size.L, size.N, K, false, B_code, stream);
    }
    if (resources.B_col_major) {
        launch_init_test_data(resources.B_col_major, size.L, size.N, K, true, B_code, stream);
    }

    // K-major inputs required to initialize expected data.
    assert(resources.A_row_major);
    assert(resources.B_col_major);
    if (resources.C_expected_row_major) {
        using G = GemmEx<A_row_major_flag | C_row_major_flag, Ctype, ABtype, Ctype>;
        G::run(resources.cublasH, size, resources.A_row_major, resources.B_col_major, resources.C_expected_row_major);
    }
    if (resources.C_expected_col_major) {
        using G = GemmEx<A_row_major_flag, Ctype, ABtype, Ctype>;
        G::run(resources.cublasH, size, resources.A_row_major, resources.B_col_major, resources.C_expected_col_major);
    }
}

template <typename Ctype, typename ABtype>
TestResult gemm_case_visitor_impl(
        const GemmCaseT<Ctype, ABtype>& gemm_case,
        const GemmTestResourcesUnion& resources_union,
        GemmSize size,
        TestCheckMode check_mode)
{
    using namespace ::sporkbench::sporkbench_test;
    using Resources = GemmTestResourcesT<Ctype, ABtype>;
    const Resources resources = std::get<Resources>(resources_union);

    const cudaStream_t stream = 0;

    if (check_mode == TestCheckMode::none && gemm_case.test_correctness_only) {
        return TestResult::passed_0_flops;
    }

    // Fill output C matrices with garbage.
    if (check_mode != TestCheckMode::none) {
        cudaMemsetAsync(resources.C_test, 0xDD, sizeof(resources.C_test[0]) * size.L * size.M * size.N);
    }

    const double flops = run_gemm_case_impl(gemm_case, resources, size, stream);

    bool passed = true;
    if (check_mode != TestCheckMode::none) {
        const bool row_major = bool(gemm_case.flags & C_row_major_flag);
        const bool exact = (check_mode == TestCheckMode::exact);
        const Ctype* C = row_major ? resources.C_expected_row_major : resources.C_expected_col_major;
        assert(C);
        passed = launch_device_compare_tensor(
                size, gemm_case.proc_name, resources.C_test, C, row_major, exact, stream);
    }

    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    TestResult result{};
    result.flops = flops;
    result.passed = passed;
    return result;
}

void init_test_data(GemmTestResourcesUnion resources, GemmSize size, TestDataCode A_code, TestDataCode B_code)
{
    auto visitor = [&] (auto typed_resources)
    {
        init_test_data_impl(typed_resources, size, A_code, B_code);
    };
    return std::visit(visitor, resources);
}

TestResult run_gemm_case(
        GemmCaseUnion gemm_case, GemmTestResourcesUnion resources, GemmSize size, TestCheckMode check_mode)
{
    auto visitor = [&] (const auto& gemm_case)
    {
        return gemm_case_visitor_impl(gemm_case, resources, size, check_mode);
    };
    return std::visit(visitor, gemm_case);
}

void init_test_data(const GemvTestResources& resources, GemvSize size, TestDataCode A_code, TestDataCode B_code)
{
    using namespace ::sporkbench::sporkbench_test;
    const int L = 1;
    const cudaStream_t stream = 0;
    launch_init_test_data(resources.A, L, size.M, size.K, true, A_code, stream);
    launch_init_test_data(resources.x, L, 1, size.K, true, B_code, stream);
    run_cublas_gemv(resources.cublasH, size, resources.A, resources.x, resources.y_expected);
}

TestResult run_gemv_case(
        const GemvCase& gemv_case, const GemvTestResources& resources, GemvSize size, TestCheckMode check_mode)
{
    using namespace ::sporkbench::sporkbench_test;
    const cudaStream_t stream = 0;

    if (check_mode == TestCheckMode::none && gemv_case.test_correctness_only) {
        return TestResult::passed_0_flops;
    }

    // Fill output y_test with garbage.
    if (check_mode != TestCheckMode::none) {
        cudaMemsetAsync(resources.y_test, 0xDD, sizeof(resources.y_test[0]) * size.M);
    }

    const double flops = run_gemv_case_impl(gemv_case, resources, size, stream);

    bool passed = true;
    if (check_mode != TestCheckMode::none) {
        const bool exact = (check_mode == TestCheckMode::exact);
        passed = launch_device_compare_tensor(
                size, gemv_case.proc_name, resources.y_test, resources.y_expected, true, exact, stream);
    }

    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    TestResult result{};
    result.flops = flops;
    result.passed = passed;
    return result;
}

template <typename T_type, typename L_type, int Hdim, bool Causal>
void init_test_data_impl(
        const AttnFwdTestResourcesT<T_type, L_type, Hdim, Causal>& resources, AttnFwdSize size,
        TestDataCode Q_code, TestDataCode K_code, TestDataCode V_code)
{
    using namespace ::sporkbench::sporkbench_test;
    const auto qo_heads = size.KV_Heads * size.Groups;
    const auto kv_heads = size.KV_Heads;
    const cudaStream_t stream{};
    assert(size.Hdim == Hdim);
    launch_init_test_data(resources.d_Q, size.Batch * qo_heads, size.SeqLen, Hdim, true, Q_code, stream);
    launch_init_test_data(resources.d_K, size.Batch * kv_heads, size.SeqLen, Hdim, true, K_code, stream);
    launch_init_test_data(resources.d_V, size.Batch * kv_heads, size.SeqLen, Hdim, true, V_code, stream);

    // TODO use cuDNN or something so we don't rely on H100.
    kittens_mha_h100::attention_forward(
        size.Batch, size.KV_Heads, size.Groups, size.SeqLen, Hdim, Causal,
        resources.d_O_expected, resources.d_l_vec_expected, resources.d_Q, resources.d_K, resources.d_V,
        stream);
}

template <typename T_type, typename L_type, int Hdim, bool Causal>
TestResult attn_fwd_case_visitor_impl(
        const AttnFwdCaseT<T_type, L_type, Hdim, Causal>& attn_case,
        const AttnFwdTestResourcesUnion& resources_union,
        AttnFwdSize size,
        TestCheckMode check_mode)
{
    if (check_mode == TestCheckMode::none && attn_case.test_correctness_only) {
        return TestResult::passed_0_flops;
    }

    using namespace ::sporkbench::sporkbench_test;
    using Resources = AttnFwdTestResourcesT<T_type, L_type, Hdim, Causal>;
    const Resources resources = std::get<Resources>(resources_union);
    const auto qo_heads = size.KV_Heads * size.Groups;

    const cudaStream_t stream = 0;

    // Fill output matrices with garbage.
    if (check_mode != TestCheckMode::none) {
        cudaMemsetAsync(
                resources.d_O_test, 0xDD,
                sizeof(resources.d_O_test[0]) * size.Batch * qo_heads * size.SeqLen * Hdim,
                stream);
        cudaMemsetAsync(
                resources.d_l_vec_test, 0xDD,
                sizeof(resources.d_l_vec_test[0]) * size.Batch * qo_heads * size.SeqLen,
                stream);
    }

    const double flops = run_attn_fwd_case_impl(attn_case, resources, size, stream);

    bool passed = true;
    if (check_mode != TestCheckMode::none) {
        const bool exact = (check_mode == TestCheckMode::exact);
        AttnStatsVectorSize l_vec_size{};
        l_vec_size.Batch = size.Batch;
        l_vec_size.KV_Heads = size.KV_Heads;
        l_vec_size.Groups = size.Groups;
        l_vec_size.SeqLen = size.SeqLen;
        l_vec_size.Hdim = size.Hdim;
        passed &= launch_device_compare_tensor(
                size, attn_case.proc_name,
                resources.d_O_test, resources.d_O_expected,
                true, exact, stream);
        passed &= launch_device_compare_tensor(
                l_vec_size, attn_case.proc_name,
                resources.d_l_vec_test, resources.d_l_vec_expected,
                true, exact, stream);
    }

    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    TestResult result{};
    result.flops = flops;
    result.passed = passed;
    return result;
}


void init_test_data(
        const AttnFwdTestResourcesUnion& resources, AttnFwdSize size,
        TestDataCode Q_code, TestDataCode K_code, TestDataCode V_code)
{
    auto visitor = [&] (auto typed_resources)
    {
        init_test_data_impl(typed_resources, size, Q_code, K_code, V_code);
    };
    return std::visit(visitor, resources);
}

TestResult run_attn_fwd_case(
        AttnFwdCaseUnion attn_fwd_case, AttnFwdTestResourcesUnion resources, AttnFwdSize size, TestCheckMode check_mode)
{
    auto visitor = [&] (auto typed_case)
    {
        return attn_fwd_case_visitor_impl(typed_case, resources, size, check_mode);
    };
    return std::visit(visitor, attn_fwd_case);
}

}  // end namespace sporkbench
