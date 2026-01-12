#pragma once

#define exo_f16 __half
#define exo_bf16 __nv_bfloat16

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <variant>
#include <vector>

namespace sporkbench {

inline const char* case_type_name(__nv_bfloat16)
{
    return "bf16";
}
inline const char* case_type_name(__half)
{
    return "f16";
}
inline const char* case_type_name(float)
{
    return "f32";
}

enum class CudaArch
{
    Sm80,
    Sm90a,
    Sm100a,
};

inline const char* arch_name(CudaArch arch)
{
    switch (arch) {
      case CudaArch::Sm80:
        return "sm_80";
      case CudaArch::Sm90a:
        return "sm_90a";
      case CudaArch::Sm100a:
        return "sm_100a";
    }
    return "sm_XXX";
}

inline bool cuda_arch_supports(CudaArch cuda_arch, int cuda_cc_major, int cuda_cc_minor)
{
    switch (cuda_arch) {
      case CudaArch::Sm80:
        return cuda_cc_major >= 8;
      case CudaArch::Sm90a:
        return cuda_cc_major == 9 && cuda_cc_minor == 0;
      case CudaArch::Sm100a:
        return cuda_cc_major == 10 && cuda_cc_minor == 0;
    }
    return false;
}

struct GemmSize
{
    int L, M, N, K_split, K_cluster;

    constexpr size_t C_col_major_index(int batch, int m, int n) const
    {
        return size_t(batch) * M * N + size_t(n) * M + m;
    }
    constexpr size_t C_row_major_index(int batch, int m, int n) const
    {
        return size_t(batch) * M * N + size_t(m) * N + n;
    }
};

struct GemvSize
{
    int M, K;
};

template <typename Ctype, typename ABtype>
using GemmRunT = void(*)(cublasHandle_t cublasH, GemmSize size, const ABtype* A, const ABtype* B, Ctype* C);

typedef void (*GemvRun)(cublasHandle_t cublasH, GemvSize size, const float* A, const float* x, float* y);

constexpr int A_row_major_flag = 1;
constexpr int B_row_major_flag = 2;
constexpr int C_row_major_flag = 4;

template <typename Ctype, typename ABtype>
struct GemmCaseT
{
    CudaArch cuda_arch;
    const char* json_name;
    const char* proc_name;
    GemmRunT<Ctype, ABtype> run_function;
    int flags;
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

    bool supports_split_k() const
    {
        return K_split_max > 1;
    }

    static const char* ab_type_name()
    {
        return case_type_name(ABtype{});
    }

    static const char* c_type_name()
    {
        return case_type_name(Ctype{});
    }
};

using GemmCase = GemmCaseT<float, float>;
using GemmCase_f32_f32 = GemmCaseT<float, float>;
using GemmCase_f32_f16 = GemmCaseT<float, __half>;
using GemmCase_f16_f16 = GemmCaseT<__half, __half>;

using GemmCaseUnion = std::variant<
        GemmCase_f32_f32,
        GemmCase_f32_f16,
        GemmCase_f16_f16>;

// These are supposed to be generated from the user's JSON files.
// Note the arg is just an unused dummy object to distinguish overloads.
const std::vector<GemmCase_f32_f32>& get_user_cases(const GemmCase_f32_f32&);
const std::vector<GemmCase_f32_f16>& get_user_cases(const GemmCase_f32_f16&);
const std::vector<GemmCase_f16_f16>& get_user_cases(const GemmCase_f16_f16&);
// sporkbench_builtin_cases.cu
const std::vector<GemmCase_f32_f32>& get_builtin_cases(const GemmCase_f32_f32&);
const std::vector<GemmCase_f32_f16>& get_builtin_cases(const GemmCase_f32_f16&);
const std::vector<GemmCase_f16_f16>& get_builtin_cases(const GemmCase_f16_f16&);

// TODO templatize gemv like gemm but I think no one actually cares.
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

    // For now we hard-wire all gemv kernels not to do split K.
    static constexpr int K_split_divisor = 1;
    static constexpr int K_split_max = 1;

    bool supports(GemvSize size) const
    {
        return (
            size.M <= M_max && size.M % M_divisor == 0 &&
            size.K <= K_max && size.K % K_divisor == 0
        );
    }

    bool supports_split_k() const
    {
        return false;
    }
};

using GemvCase_f32_f32 = GemvCase;

// These are supposed to be generated from the user's JSON files.
// Note the arg is just an unused dummy object to distinguish overloads.
const std::vector<GemvCase_f32_f32>& get_user_cases(const GemvCase_f32_f32&);
// sporkbench_builtin_cases.cu
const std::vector<GemvCase_f32_f32>& get_builtin_cases(const GemvCase_f32_f32&);



// sporkbench_builtin_cases.cu
void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const float* A, const float* B, float* C);
void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const __half* A, const __half* B, float* C);
void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const __half* A, const __half* B, __half* C);
void run_cublas_gemv(cublasHandle_t cublasH, GemvSize size, const float* A, const float* x, float* y);

}
