#pragma once

namespace sporkbench {

struct GemmSize
{
    int L, M, N, K_split, K_cluster;
};

struct GemvSize
{
    int M, K;
};

// A is row major (both functions), B and C are column major.
typedef void (*GemmRun)(GemmSize size, const float* A, const float* B, float* C);
typedef void (*GemvRun)(GemvSize size, const float* A, const float* x, float* y);

struct GemmCase
{
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
            size.L < L_max && size.L % L_divisor == 0 &&
            size.M < M_max && size.M % M_divisor == 0 &&
            size.N < N_max && size.N % N_divisor == 0 &&
            size.K_split < K_split_max && size.K_split % K_split_divisor == 0 &&
            size.K_cluster < K_cluster_max && size.K_cluster % K_cluster_divisor == 0
        );
    }
};

struct GemvCase
{
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
            size.M < M_max && size.M % M_divisor == 0 &&
            size.K < K_max && size.K % K_divisor == 0
        );
    }
};

extern const GemmCase user_gemm_cases[];
extern const int num_user_gemm_cases;
extern const GemmCase builtin_gemm_cases[];
extern const int num_builtin_gemm_cases;

extern const GemvCase user_gemv_cases[];
extern const int num_user_gemv_cases;
extern const GemvCase builtin_gemv_cases[];
extern const int num_builtin_gemv_cases;

}
