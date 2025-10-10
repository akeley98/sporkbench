#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <tuple>
#include <vector>

#include "sporkbench_test.hpp"
#include "sporkbench_cases.hpp"

#define CUBLAS_CHECK(x) if (auto _cublas_status = x; _cublas_status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%i cublas status %i\n", __FILE__, __LINE__, (int)_cublas_status); }

namespace sporkbench
{

static std::vector<std::tuple<int, int, int>> generate_mnk_sizes()
{
    return {{2048, 2048, 2048}, {4096, 4096, 4096}, {7680, 7680, 8192}, {2816, 768, 65536}};
}

struct AsyncDeleter
{
    cudaStream_t stream;

    void operator() (void* victim)
    {
        cudaFreeAsync(victim, stream);
    }

    std::unique_ptr<float[], AsyncDeleter> alloc(int L, int MN, int K) const
    {
        size_t sz = sizeof(float) * size_t(L) * size_t(MN) * size_t(K);
        void* ptr = nullptr;
        cudaMallocAsync(&ptr, sz, stream);
        if (!ptr and sz > 0) {
            cudaError_t err = cudaGetLastError();
            throw std::runtime_error(cudaGetErrorString(err));
        }
        return std::unique_ptr<float[], AsyncDeleter>(static_cast<float*>(ptr), *this);
    }
};

int Main(int argc, char** argv)
{
    cudaSetDevice(0);

    int cuda_cc_major{}, cuda_cc_minor{};
    cudaDeviceGetAttribute(&cuda_cc_major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&cuda_cc_minor, cudaDevAttrComputeCapabilityMinor, 0);
    const bool is_h100 = cuda_cc_major == 9 && cuda_cc_minor == 0;
    fprintf(stderr, "is_h100: %i\n", is_h100);

    cudaStream_t stream{};
    cublasHandle_t cublasH{};
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_TF32_TENSOR_OP_MATH));

    bool all_passed = true;

    for (std::tuple<int, int, int> mnk: generate_mnk_sizes()) {
        auto [M, N, K] = mnk;
        AsyncDeleter deleter{stream};
        for (int L = 1; L < 5; L *= 4) {
            printf("L = %i, MNK = [%i, %i, %i]\n", L, M, N, K);
            std::unique_ptr<float[], AsyncDeleter> unique_A = deleter.alloc(L, M, K);
            std::unique_ptr<float[], AsyncDeleter> unique_B = deleter.alloc(L, N, K);
            std::unique_ptr<float[], AsyncDeleter> unique_C_test = deleter.alloc(L, M, N);
            std::unique_ptr<float[], AsyncDeleter> unique_C_expected = deleter.alloc(L, M, N);

            double best_flops = 0;
            int winner_K_split = 0;
            const GemmCase* p_winner_user_case = nullptr;

            auto gemm_case_helper = [&] (const GemmCase& gemm_case, bool is_builtin, int K_split_if_winner)
            {
                if (!cuda_arch_supports(gemm_case.cuda_arch, cuda_cc_major, cuda_cc_minor)) {
                    // Skip unsupported architectures.
                    if (false) {
                        printf("    Skipped: %s\n", gemm_case.proc_name);
                    }
                    return;  // Exit lambda
                }

                for (int K_split = 1; K_split <= 16; K_split *= 2) {
                    if (K_split_if_winner && K_split != K_split_if_winner) {
                        continue;
                    }
                    GemmSize size{};
                    size.L = L;
                    size.M = M;
                    size.N = N;
                    size.K_split = K_split;
                    size.K_cluster = K / K_split;
                    if (size.K_split * size.K_cluster != K) {
                        // We only support exact divisibilty for K_split for now.
                        break;
                    }
                    const bool is_winner = K_split_if_winner != 0;
                    const int num_trials = 25;
                    if (gemm_case.supports(size)) {
                        int color_code = is_winner ? 33 : is_builtin ? 32 : K_split > 1 ? 36 : 0;
                        TestResult result = run_gemm_case(
                                gemm_case, cublasH, size,
                                unique_A.get(), unique_B.get(), unique_C_test.get(), unique_C_expected.get(),
                                true, num_trials);
                        printf("%8.3f \x1b[%imTFLOPS\x1b[0m; K/%i, %s%s\n",
                            result.flops / 1e12, color_code, K_split, gemm_case.proc_name,
                            is_winner ? " (winner)" : "");
                        all_passed &= result.passed;

                        if (!is_builtin && result.passed && best_flops < result.flops) {
                            best_flops = result.flops;
                            p_winner_user_case = &gemm_case;
                            winner_K_split = K_split;
                        }
                    }
                }
            };

            for (int case_i = 0; case_i < num_user_gemm_cases; ++case_i) {
                gemm_case_helper(user_gemm_cases[case_i], false, 0);
            }
            for (int case_i = 0; case_i < num_builtin_gemm_cases; ++case_i) {
                gemm_case_helper(builtin_gemm_cases[case_i], true, 0);
            }
            if (p_winner_user_case) {
                gemm_case_helper(*p_winner_user_case, false, winner_K_split);
            }
        }
    }

    if (all_passed) {
        printf("All tests passed.\n");
    }
    else {
        printf("\x1b[31m\x1b[1mFAILED:\x1b[0m Not all test cases passed!\n");
    }

    cublasDestroy(cublasH);
    return all_passed ? 0 : 1;
}

}  // end namespace

int main(int argc, char** argv)
{
    return sporkbench::Main(argc, argv);
}
