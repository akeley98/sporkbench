#include "sporkbench_test.hpp"

#include <bit>
#include <cassert>
#include <cublas_v2.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <vector>

namespace sporkbench {

namespace sporkbench_test {

enum class TestDataCode
{
    random = 0,
    batch_index_identity = 1,
    tiled_numbers = 2,
    signs_only = 3,
};

// Copied pseudo random number generation code.
// http://www.jcgt.org/published/0009/03/02/
// Hash Functions for GPU Rendering, Mark Jarzynski, Marc Olano, NVIDIA
__device__ uint64_t pcg3d(uint32_t x, uint32_t y, uint32_t z)
{
    x = x*1664525u + 1013904223u;
    y = y*1664525u + 1013904223u;
    z = z*1664525u + 1013904223u;

    x += y*z;
    y += z*x;
    z += x*y;

    x ^= x >> 16u;
    y ^= y >> 16u;
    z ^= z >> 16u;

    x += y*z;
    y += z*x;
    z += x*y;

    return x ^ uint64_t(y) << 12u ^ uint64_t(z) << 24u;
}

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
                    value = T((k % 64) + 100 * (mn % 64));
                    break;
                  case TestDataCode::signs_only:
                    {
                        // uniform choice between -1, 0, 1.
                        value = T(int(pcg3d(k, mn, z + 20010106) % 3) - 1);
                    }
                    break;
                  case TestDataCode::random:
                  default:
                    {
                        const auto randbits = pcg3d(k, mn, z + 20010106);
                        if (randbits % 100'000u == 0) {
                            // 1 in 100'000 chance of a "big" value (1000).
                            // This greatly reduces the chance that a genuine bug is mistaken for fp error.
                            value = T(1000);
                        }
                        else if (randbits % 4u != 0u) {
                            value = T(0);  // 75% chance of a 0
                        }
                        else {
                            // 25% chance of random value [0, 1], biased towards small numbers.
                            value = T((pcg3d(k, mn, 19980724) % 1'000'000) * 1e-6f);
                            value = (value * value) * (value * value);
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
    dim3 grid{(minor_extent + 255u) / 256u, major_extent, batch_size};
    dim3 block{256, 1, 1};
    device_init_test_data<<<grid, block, 0, stream>>>(d_tensor, batch_size, MN, K, k_major, code);
}


__global__ void device_compare_tensor_test_init_bitfield(unsigned long long* d_bitfield)
{
    *d_bitfield = UINT64_MAX;
}

// Requires that *d_bitfield is initialized to UINT64_MAX.
// Compare the two equal-sized matrices and, if any comparison failures, put the coordinates of the wrong value
// into *d_bitfield, packed as its linear_index into the d_expected array.
// d_expected is always column major. d_test is described by the test_row_major flag.
template <typename Test, typename Expected>
__global__ void device_compare_tensor_test(GemmSize size, const Test* d_test, const Expected* d_expected,
                                           bool test_row_major, bool exact, unsigned long long* d_bitfield)
{
    uint32_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t tid_z = threadIdx.z + blockIdx.z * blockDim.z;
    for (uint32_t batch = tid_z; batch < size.L; batch += blockDim.z * gridDim.z) {
        for (uint32_t m = tid_y; m < size.M; m += blockDim.y * gridDim.y) {
            for (uint32_t n = tid_x; n < size.N; n += blockDim.x * gridDim.x) {
                const size_t expected_linear_index = size.C_col_major_index(batch, m, n);
                size_t test_linear_index = expected_linear_index;
                if (test_row_major) {
                    test_linear_index = size.C_row_major_index(batch, m, n);
                }
                bool correct;
                if (exact) {
                    correct = d_test[test_linear_index] == d_expected[expected_linear_index];
                }
                else {
                    float f_test = float(d_test[test_linear_index]);
                    float f_expected = float(d_expected[expected_linear_index]);
                    correct = f_test * f_expected >= 0.0f;  // Sign error, or inf/nan if wrong
                    if (correct) {
                        f_test = fabsf(f_test);
                        f_expected = fabsf(f_expected);
                        const float Min = fminf(f_test, f_expected);
                        const float Max = fmaxf(f_test, f_expected);
                        correct = Max == 0 || Max / Min < (1.0f + 1/128.0f);
                    }
                }
                if (!correct) {
                    atomicMin(d_bitfield, (unsigned long long)expected_linear_index);
                }
            }
        }
    }
}

template <typename Test, typename Expected>
__device__ void print_tensor_neighborhood(GemmSize size, const Test* d_test, const Expected* d_expected,
                                          bool test_row_major, uint32_t batch, uint32_t m, uint32_t n)
{
    uint32_t m_min = m < 2 ? 0u : m - 2;
    uint32_t m_max = m + 2 >= size.M ? size.M - 1u : m + 2;
    uint32_t n_min = n < 2 ? 0u : n - 2;
    uint32_t n_max = n + 2 >= size.N ? n - 1u : n + 2;

    for (uint32_t cm = m_min; cm <= m_max; cm++) {
        for (uint32_t cn = n_min; cn <= n_max; cn++) {
            if (cn == n && cm == m) {
                printf("\x1b[1m");
            }
            const size_t expected_linear_index = size.C_col_major_index(batch, m, n);
            size_t test_linear_index = expected_linear_index;
            if (test_row_major) {
                test_linear_index = size.C_row_major_index(batch, m, n);
            }
            printf("[%6g, %5g]  ",
                static_cast<double>(d_test[test_linear_index]),
                static_cast<double>(d_expected[expected_linear_index]));
            if (cn == n && cm == m) {
                printf("\x1b[0m");
            }
        }
        printf("\n");
    }
}



// Print info on wrong value from function above.
template <typename Test, typename Expected>
__global__ void device_compare_tensor_test_print(GemmSize size, const Test* d_test, const Expected* d_expected,
                                                 bool test_row_major, unsigned long long* d_bitfield)
{
    unsigned long long expected_linear_index = *d_bitfield;
    if (expected_linear_index != UINT64_MAX) {
        const uint32_t batch = expected_linear_index / (size.M * size.N);
        const uint32_t n = (expected_linear_index / size.M) % size.N;
        const uint32_t m = expected_linear_index % size.M;
        unsigned long long test_linear_index = expected_linear_index;
        if (test_row_major) {
            test_linear_index = size.C_row_major_index(batch, m, n);
        }
        const double f_test = static_cast<double>(d_test[test_linear_index]);
        const double f_expected = static_cast<double>(d_expected[expected_linear_index]);
        printf("\x1b[1m[batch=%u, m=%u, n=%u]\x1b[0m %g != %g (test != expected)\n", batch, m, n, f_test, f_expected);

        print_tensor_neighborhood(size, d_test, d_expected, test_row_major, batch, m, n);
    }
}

template <typename Test, typename Expected>
bool launch_device_compare_tensor(
        GemmSize size, const char* proc_name, const Test* d_test, const Expected* d_expected, bool test_row_major,
        bool exact, cudaStream_t stream)
{
    dim3 grid(unsigned(size.N + 127) / 128u, unsigned(size.M), unsigned(size.L));
    dim3 block(128, 1, 1);
    unsigned long long* d_bitfield = 0;
    cudaMallocAsync(&d_bitfield, sizeof(*d_bitfield), stream);
    if (!d_bitfield) {
        fprintf(stderr, "Alloc of d_bitfield failed: %s:%i\n", __FILE__, __LINE__);
        return false;
    }
    device_compare_tensor_test_init_bitfield<<<1, 1, 0, stream>>>(d_bitfield);
    device_compare_tensor_test<<<grid, block, 0, stream>>>(size, d_test, d_expected, test_row_major, exact, d_bitfield);
    unsigned long long h_bitfield;
    cudaMemcpy(&h_bitfield, d_bitfield, sizeof(h_bitfield), cudaMemcpyDeviceToHost);
    if (h_bitfield != UINT64_MAX) {
        printf("\x1b[31m\x1b[1mFAILED:\x1b[0m %s, L=%i, MNK=[%i, %i, %i], K_split=%i\n",
            proc_name, size.L, size.M, size.N, size.K_cluster * size.K_split, size.K_split);
        fflush(stdout);
        device_compare_tensor_test_print<<<1, 1, 0, stream>>>(size, d_test, d_expected, test_row_major, d_bitfield);
        cudaStreamSynchronize(stream);  // flush stdout.
    }
    cudaFreeAsync(d_bitfield, stream);
    return h_bitfield == UINT64_MAX;
}


TestResult run_gemm_case_test_data(
        const GemmCase& gemm_case, cublasHandle_t cublasH, GemmSize size,
        float* A, float* B, float* C_test, float* C_expected,
        TestDataCode A_code, TestDataCode B_code, int test_count, cudaStream_t stream)
{
    const uint32_t L = uint32_t(size.L);
    const uint32_t M = uint32_t(size.M);
    const uint32_t N = uint32_t(size.N);
    const uint32_t K = uint32_t(size.K_split * size.K_cluster);

    // For our use of cublas, we have A row major, B and C column major.
    // Both A and B are K-major per Nvidia's terminology.
    launch_init_test_data(A, L, M, K, true, A_code, stream);
    launch_init_test_data(B, L, N, K, true, B_code, stream);
    run_cublas_gemm(cublasH, size, A, B, C_expected);

    std::vector<float> test_times(test_count);
    std::vector<cudaEvent_t> test_events(test_count + 1);
    auto new_event = [stream]
    {
        cudaEvent_t event{};
        if (const cudaError_t err = cudaEventCreate(&event)) {
            throw std::runtime_error("cudaEventCreate failed\n");
        }
        cudaEventRecord(event, stream);
        return event;
    };

    const bool A_row_major = bool(gemm_case.flags & A_row_major_flag);
    const bool B_row_major = bool(gemm_case.flags & B_row_major_flag);
    launch_init_test_data(A, L, M, K, A_row_major, A_code, stream);   // K-major == A row major
    launch_init_test_data(B, L, N, K, !B_row_major, B_code, stream);  // K-major == B column major
    for (int test_i = 0; test_i < test_count; ++test_i) {
        if (test_i == 0) {
            test_events[0] = new_event();
        }
        gemm_case.run_function(cublasH, size, A, B, C_test);
        test_events[test_i + 1] = new_event();
    }
    const bool exact = A_code == TestDataCode::signs_only && B_code == TestDataCode::signs_only && K <= 4096;
    const bool test_row_major = bool(gemm_case.flags & C_row_major_flag);
    const bool passed = launch_device_compare_tensor(
            size, gemm_case.proc_name, C_test, C_expected, test_row_major,
            exact, stream);

    for (int test_i = 0; test_i < test_count; ++test_i) {
        cudaEventElapsedTime(&test_times[test_i], test_events[test_i], test_events[test_i + 1]);
        cudaEventDestroy(test_events[test_i]);
    }
    cudaEventDestroy(test_events[test_count]);
    std::sort(&test_times[0], &test_times[test_count]);
    const double median_ms = test_times[test_count / 2];
    const double flops = double(L) * M * N * K * 2000.0 / median_ms;
    return TestResult{passed, flops};
}

}  // end namespace sporkbench_test

TestResult run_gemm_case(
        const GemmCase& gemm_case, cublasHandle_t cublasH, GemmSize size,
        float* A, float* B, float* C_test, float* C_expected,
        bool warmup, int num_trials)
{
    using namespace ::sporkbench::sporkbench_test;
    const cudaStream_t stream = 0;
    assert(num_trials > 0);

    // Fill output C matrices with garbage.
    cudaMemsetAsync(C_test, 0xDD, sizeof(C_test[0]) * size.L * size.M * size.N);

    if (warmup) {
        run_gemm_case_test_data(
                gemm_case, cublasH, size, A, B, C_test, C_expected,
                TestDataCode::batch_index_identity, TestDataCode::tiled_numbers, 1, stream);
        run_gemm_case_test_data(
                gemm_case, cublasH, size, A, B, C_test, C_expected,
                TestDataCode::tiled_numbers, TestDataCode::batch_index_identity, 1, stream);
        run_gemm_case_test_data(
                gemm_case, cublasH, size, A, B, C_test, C_expected,
                TestDataCode::signs_only, TestDataCode::signs_only, 1, stream);
    }
    const TestResult result = run_gemm_case_test_data(
            gemm_case, cublasH, size, A, B, C_test, C_expected,
            TestDataCode::random, TestDataCode::random, num_trials, stream);
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return result;
}

}  // end namespace sporkbench
