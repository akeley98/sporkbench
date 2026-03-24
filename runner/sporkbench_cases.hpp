#pragma once

#define exo_f16 __half
#define exo_bf16 __nv_bfloat16
#define exo_e4m3 __nv_fp8_e4m3
#define exo_e5m2 __nv_fp8_e5m2

#include <cublas_v2.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <variant>
#include <vector>

namespace sporkbench {

static_assert(!std::is_same_v<exo_e4m3, exo_e5m2>);

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
inline const char* case_type_name(exo_e4m3)
{
    return "e4m3";
}
inline const char* case_type_name(exo_e5m2)
{
    return "e5m2";
}

enum class CudaArch
{
    Sm80,
    Sm90a,
    Sm100a,  // TODO
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

struct AttnFwdSize
{
    int Batch, KV_Heads, Groups, Hdim, SeqLen;
    // Q[Batch, KV_Heads, Groups, SeqLen, Hdim]; `KV_Heads * Groups` is the total number of heads
    // K[Batch, KV_Heads, SeqLen, Hdim]; `KV_Heads` is the total number of heads
    // V[Batch, KV_Heads, SeqLen, Hdim]; `KV_Heads` is the total number of heads
    // O[Batch, KV_Heads, Groups, SeqLen, Hdim]; `KV_Heads * Groups` is the total number of heads
    // l_vec[Batch, KV_Heads, Groups, SeqLen]; `KV_Heads * Groups` is the total number of heads
    //
    // Rightmost stride is 1.
};

template <typename Ctype, typename ABtype>
using GemmRunT = void(*)(cublasHandle_t cublasH, GemmSize size, const ABtype* A, const ABtype* B, Ctype* C);

typedef void (*GemvRun)(cublasHandle_t cublasH, GemvSize size, const float* A, const float* x, float* y);

template <typename T_type, typename L_type, int Hdim>
using AttnFwdRunT = void(*)(AttnFwdSize size, T_type* O, L_type* l_vec, const T_type* Q, const T_type* K, const T_type* V);

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
using GemmCase_f32_bf16 = GemmCaseT<float, exo_bf16>;
using GemmCase_f32_e4m3 = GemmCaseT<float, exo_e4m3>;
using GemmCase_f32_e5m2 = GemmCaseT<float, exo_e5m2>;

using GemmCaseUnion = std::variant<
        GemmCase_f32_f32,
        GemmCase_f32_f16,
        GemmCase_f16_f16,
        GemmCase_f32_bf16,
        GemmCase_f32_e4m3,
        GemmCase_f32_e5m2>;

// These are supposed to be generated from the user's JSON files.
// Note the arg is just an unused dummy object to distinguish overloads.
const std::vector<GemmCase_f32_f32>& get_user_cases(const GemmCase_f32_f32&);
const std::vector<GemmCase_f32_f16>& get_user_cases(const GemmCase_f32_f16&);
const std::vector<GemmCase_f16_f16>& get_user_cases(const GemmCase_f16_f16&);
const std::vector<GemmCase_f32_bf16>& get_user_cases(const GemmCase_f32_bf16&);
const std::vector<GemmCase_f32_e4m3>& get_user_cases(const GemmCase_f32_e4m3&);
const std::vector<GemmCase_f32_e5m2>& get_user_cases(const GemmCase_f32_e5m2&);
// sporkbench_builtin_cases.cu
const std::vector<GemmCase_f32_f32>& get_builtin_cases(const GemmCase_f32_f32&);
const std::vector<GemmCase_f32_f16>& get_builtin_cases(const GemmCase_f32_f16&);
const std::vector<GemmCase_f16_f16>& get_builtin_cases(const GemmCase_f16_f16&);
const std::vector<GemmCase_f32_bf16>& get_builtin_cases(const GemmCase_f32_bf16&);
const std::vector<GemmCase_f32_e4m3>& get_builtin_cases(const GemmCase_f32_e4m3&);
const std::vector<GemmCase_f32_e5m2>& get_builtin_cases(const GemmCase_f32_e5m2&);

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

template <typename T_type, typename L_type, int Hdim_, bool Causal_>
struct AttnFwdCaseT
{
    static constexpr bool Causal = Causal_;
    static constexpr int Hdim = Hdim_;
    static constexpr int K_split_divisor = 1;
    static constexpr int K_split_max = 1;

    CudaArch cuda_arch;
    const char* json_name;
    const char* proc_name;
    AttnFwdRunT<T_type, L_type, Hdim_> run_function;
    int Batch_divisor;
    int Batch_max;
    int KV_Heads_divisor;
    int KV_Heads_max;
    int Groups_divisor;
    int Groups_max;
    int SeqLen_divisor;
    int SeqLen_max;

    bool supports(AttnFwdSize size) const
    {
        return (
            size.Batch <= Batch_max &&
            size.Batch % Batch_divisor == 0 &&
            size.KV_Heads <= KV_Heads_max &&
            size.KV_Heads % KV_Heads_divisor == 0 &&
            size.Hdim == Hdim &&
            size.SeqLen <= SeqLen_max &&
            size.SeqLen % SeqLen_divisor == 0
        );
    }

    bool supports_split_k() const
    {
        return false;
    }

    static const char* t_type_name()
    {
        return case_type_name(T_type{});
    }

    static const char* l_type_name()
    {
        return case_type_name(L_type{});
    }
};

using AttnFwdCase_bf16_f32_64 = AttnFwdCaseT<exo_bf16, float, 64, false>;
using AttnFwdCase_bf16_f32_128 = AttnFwdCaseT<exo_bf16, float, 128, false>;
using AttnFwdCase_bf16_f32_64_causal = AttnFwdCaseT<exo_bf16, float, 64, true>;
using AttnFwdCase_bf16_f32_128_causal = AttnFwdCaseT<exo_bf16, float, 128, true>;

using AttnFwdCaseUnion = std::variant<
        AttnFwdCase_bf16_f32_64,
        AttnFwdCase_bf16_f32_128,
        AttnFwdCase_bf16_f32_64_causal,
        AttnFwdCase_bf16_f32_128_causal>;

// These are supposed to be generated from the user's JSON files.
// Note the arg is just an unused dummy object to distinguish overloads.
const std::vector<AttnFwdCase_bf16_f32_64>& get_user_cases(const AttnFwdCase_bf16_f32_64&);
const std::vector<AttnFwdCase_bf16_f32_128>& get_user_cases(const AttnFwdCase_bf16_f32_128&);
const std::vector<AttnFwdCase_bf16_f32_64_causal>& get_user_cases(const AttnFwdCase_bf16_f32_64_causal&);
const std::vector<AttnFwdCase_bf16_f32_128_causal>& get_user_cases(const AttnFwdCase_bf16_f32_128_causal&);
// sporkbench_builtin_cases.cu
const std::vector<AttnFwdCase_bf16_f32_64>& get_builtin_cases(const AttnFwdCase_bf16_f32_64&);
const std::vector<AttnFwdCase_bf16_f32_128>& get_builtin_cases(const AttnFwdCase_bf16_f32_128&);
const std::vector<AttnFwdCase_bf16_f32_64_causal>& get_builtin_cases(const AttnFwdCase_bf16_f32_64_causal&);
const std::vector<AttnFwdCase_bf16_f32_128_causal>& get_builtin_cases(const AttnFwdCase_bf16_f32_128_causal&);


// sporkbench_builtin_cases.cu
void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const float* A, const float* B, float* C);
void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const __half* A, const __half* B, float* C);
void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const __half* A, const __half* B, __half* C);
void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const __nv_bfloat16* A, const __nv_bfloat16* B, float* C);
void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const exo_e4m3* A, const exo_e4m3* B, float* C);
void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const exo_e5m2* A, const exo_e5m2* B, float* C);
void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const float* A, const float* B, float* C);

void run_cublas_gemv(cublasHandle_t cublasH, GemvSize size, const float* A, const float* x, float* y);

}
