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

typedef void (*GemmRun)(GemmSize size, const float* A, const float* B, float* C);
typedef void (*GemvRun)(GemvSize size, const float* A, const float* x, float* y);

struct GemmCase
{
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
            size.K_divisor < K_divisor_max && size.K_divisor % K_divisor_divisor == 0
        );
    }
};

}
