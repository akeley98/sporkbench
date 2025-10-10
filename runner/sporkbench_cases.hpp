#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace sporkbench {

enum class CudaArch
{
    Sm80,
    Sm90a,
};

inline bool cuda_arch_supports(CudaArch cuda_arch, int cuda_cc_major, int cuda_cc_minor)
{
    switch (cuda_arch) {
      case CudaArch::Sm80:
        return cuda_cc_major >= 8;
      case CudaArch::Sm90a:
        return cuda_cc_major == 9 && cuda_cc_minor == 0;
    }
    return false;
}

struct GemmSize
{
    int L, M, N, K_split, K_cluster;
};

struct GemvSize
{
    int M, K;
};

// A is row major (both functions), B and C are column major.
typedef void (*GemmRun)(cublasHandle_t cublasH, GemmSize size, const float* A, const float* B, float* C);
typedef void (*GemvRun)(cublasHandle_t cublasH, GemvSize size, const float* A, const float* x, float* y);

struct GemmCase
{
    CudaArch cuda_arch;
    const char* json_name;
    const char* proc_name;
    GemmRun run_function;
    int L_divisor;
    int L_max;
    int M_divisor;
    int M_max;
    int N_divisor;
    int N_max;
    int K_split_divisor;
    int K_split_max;
    int K_cluster_divisor;
    int K_cluster_max;

    bool supports(GemmSize size) const
    {
        return (
            size.L <= L_max && size.L % L_divisor == 0 &&
            size.M <= M_max && size.M % M_divisor == 0 &&
            size.N <= N_max && size.N % N_divisor == 0 &&
            size.K_split <= K_split_max && size.K_split % K_split_divisor == 0 &&
            size.K_cluster <= K_cluster_max && size.K_cluster % K_cluster_divisor == 0
        );
    }
};

struct GemvCase
{
    CudaArch cuda_arch;
    const char* json_name;
    const char* proc_name;
    GemvRun run_function;
    int M_divisor;
    int M_max;
    int K_divisor;
    int K_max;

    bool supports(GemvSize size) const
    {
        return (
            size.M <= M_max && size.M % M_divisor == 0 &&
            size.K <= K_max && size.K % K_divisor == 0
        );
    }
};

// These are supposed to be generated from the user's JSON files.
extern const GemmCase user_gemm_cases[];
extern const int num_user_gemm_cases;
extern const GemvCase user_gemv_cases[];
extern const int num_user_gemv_cases;

// sporkbench_builtin_cases.cu
extern const GemmCase builtin_gemm_cases[];
extern const int num_builtin_gemm_cases;
extern const GemvCase builtin_gemv_cases[];
extern const int num_builtin_gemv_cases;
void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const float* A, const float* B, float* C);

}
