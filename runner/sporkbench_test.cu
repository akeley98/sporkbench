#include "sporkbench_test.hpp"

#include <bit>
#include <cassert>
#include <cublas_v2.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#include "sporkbench_pcg3d.hpp"

namespace sporkbench {

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
            const size_t expected_linear_index = size.C_col_major_index(batch, cm, cn);
            size_t test_linear_index = expected_linear_index;
            if (test_row_major) {
                test_linear_index = size.C_row_major_index(batch, cm, cn);
            }
            const double f_test = static_cast<double>(d_test[test_linear_index]);
            const double f_expected = static_cast<double>(d_expected[expected_linear_index]);
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


double run_gemm_case_impl(
        const GemmCase& gemm_case, const GemmTestResources& resources, GemmSize size, cudaStream_t stream)
{
    const uint32_t L = uint32_t(size.L);
    const uint32_t M = uint32_t(size.M);
    const uint32_t N = uint32_t(size.N);
    const uint32_t K = uint32_t(size.K_split * size.K_cluster);
    const float* A = (gemm_case.flags & A_row_major_flag) ? resources.A_row_major : resources.A_col_major;
    assert(A);
    const float* B = (gemm_case.flags & B_row_major_flag) ? resources.B_row_major : resources.B_col_major;
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

}  // end namespace sporkbench_test

void init_test_data(const GemmTestResources& resources, GemmSize size, TestDataCode A_code, TestDataCode B_code)
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
    run_cublas_gemm(resources.cublasH, size, resources.A_row_major, resources.B_col_major, resources.C_expected);
}

TestResult run_gemm_case(
        const GemmCase& gemm_case, const GemmTestResources& resources, GemmSize size, TestCheckMode check_mode)
{
    using namespace ::sporkbench::sporkbench_test;
    const cudaStream_t stream = 0;

    // Fill output C matrices with garbage.
    if (check_mode != TestCheckMode::none) {
        cudaMemsetAsync(resources.C_test, 0xDD, sizeof(resources.C_test[0]) * size.L * size.M * size.N);
    }

    const double flops = run_gemm_case_impl(gemm_case, resources, size, stream);

    bool passed = true;
    if (check_mode != TestCheckMode::none) {
        const bool test_row_major = bool(gemm_case.flags & C_row_major_flag);
        const bool exact = (check_mode == TestCheckMode::exact);
        passed = launch_device_compare_tensor(
                size, gemm_case.proc_name, resources.C_test, resources.C_expected, test_row_major, exact, stream);
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

}  // end namespace sporkbench
