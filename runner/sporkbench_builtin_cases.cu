#include "sporkbench_cases.hpp"

#include <cassert>
#include <cublas_v2.h>
#include <stdio.h>

namespace sporkbench {

#define CUBLAS_CHECK(x) if (auto _cublas_status = x; _cublas_status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%i cublas status %i\n", __FILE__, __LINE__, (int)_cublas_status); }

// This is expecting A row major, B and C column major.
void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const float* A, const float* B, float* C)
{
    assert(cublasH);
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int M = int(size.M);
    const int N = int(size.N);
    const int K = int(size.K_split * size.K_cluster);
    if (size.L == 1) {
        CUBLAS_CHECK(cublasSgemm(
                cublasH, transa, transb,
                M, N, K, &alpha,
                A, K,
                B, K,
                &beta, C, int(size.M)));
    }
    else {
        CUBLAS_CHECK(cublasSgemmStridedBatched(
                cublasH, transa, transb,
                M, N, K, &alpha,
                A, K, M * K,
                B, K, N * K,
                &beta, C, M, M * N, size.L));
    }
}

const GemmCase GemmCase::builtin_cases[] = {
  GemmCase{
    CudaArch::Sm80,
    "sporkbench_builtin_cases.cu",
    "cublas_gemm",
    run_cublas_gemm,
    A_row_major_flag,
    1, INT32_MAX,  // L
    1, INT32_MAX,  // M
    1, INT32_MAX,  // N
    1, 1,  // K_split: set to 1, so we don't sweep this parameter.
    1, INT32_MAX,  // K_cluster
  },
};
const int GemmCase::num_builtin_cases = sizeof(GemmCase::builtin_cases) / sizeof(GemmCase::builtin_cases[0]);

}  // end namespace

