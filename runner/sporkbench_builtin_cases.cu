#include "sporkbench_cases.hpp"

#include <cassert>
#include <cublas_v2.h>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <type_traits>
#include <utility>

#include "sporkbench_cutlass_Sm80.hpp"
#include "sporkbench_kittens_mha_Sm90a.hpp"

#include "pldi_Sm80_edited/exocc_Sm80_edited.h"

namespace sporkbench {

#define CUBLAS_CHECK(x) if (auto _cublas_status = x; _cublas_status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%i cublas status %i\n", __FILE__, __LINE__, (int)_cublas_status); }

template <int flags, typename Ctype, typename ABtype, typename ComputeType>
struct GemmEx
{
    static cublasComputeType_t get_compute_code(__half)
    {
        return CUBLAS_COMPUTE_16F;
    }

    static cublasComputeType_t get_compute_code(float)
    {
        return CUBLAS_COMPUTE_32F;
    }

    static cudaDataType_t get_type_code(__half)
    {
        return CUDA_R_16F;
    }

    static cudaDataType_t get_type_code(__nv_bfloat16)
    {
        return CUDA_R_16BF;
    }

    static cudaDataType_t get_type_code(float)
    {
        return CUDA_R_32F;
    }

    static cudaDataType_t get_type_code(exo_e4m3)
    {
        // Fix broken cublas fp8, we need to use tensor scaling.
        throw std::runtime_error("TODO GemmEx::get_type_code(exo_e4m3)");
        return CUDA_R_8F_E4M3;
    }

    static cudaDataType_t get_type_code(exo_e5m2)
    {
        // Fix broken cublas fp8, we need to use tensor scaling.
        throw std::runtime_error("TODO GemmEx::get_type_code(exo_e5m2)");
        return CUDA_R_8F_E5M2;
    }

    static void run(cublasHandle_t cublasH, GemmSize size, const ABtype* A, const ABtype* B, Ctype* C)
    {
        static_assert((flags & all_row_major_flags) == flags, "Unknown flag set");

        bool A_row_major;
        bool B_row_major;
        if ((flags & C_row_major_flag)) {
            // Output C of cublas is always column major.
            // So if the user wants row-major output, we have to use the transpose identity AB = (BtAt)t
            std::swap(A, B);
            std::swap(size.M, size.N);
            A_row_major = !(flags & B_row_major_flag);
            B_row_major = !(flags & A_row_major_flag);
        }
        else {
            A_row_major = (flags & A_row_major_flag);
            B_row_major = (flags & B_row_major_flag);
        }

        assert(cublasH);
        cublasOperation_t transa = A_row_major ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t transb = B_row_major ? CUBLAS_OP_T : CUBLAS_OP_N;
        const ComputeType alpha = 1.0f;
        const ComputeType beta = 0.0f;
        const int M = int(size.M);
        const int N = int(size.N);
        const int K = int(size.K_split * size.K_cluster);
        const int lda = A_row_major ? K : M;
        const int ldb = B_row_major ? N : K;
        if (size.L == 1) {
            CUBLAS_CHECK(cublasGemmEx(
                    cublasH, transa, transb,
                    M, N, K, &alpha,
                    A, get_type_code(ABtype{}), lda,
                    B, get_type_code(ABtype{}), ldb,
                    &beta, C, get_type_code(Ctype{}), M,
                    get_compute_code(ComputeType{}), CUBLAS_GEMM_DEFAULT));
        }
        else {
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                    cublasH, transa, transb,
                    M, N, K, &alpha,
                    A, get_type_code(ABtype{}), lda, M * K,
                    B, get_type_code(ABtype{}), ldb, N * K,
                    &beta, C, get_type_code(Ctype{}), M, M * N, size.L,
                    get_compute_code(ComputeType{}), CUBLAS_GEMM_DEFAULT));
        }
    }

    static GemmCaseT<Ctype, ABtype> make_case()
    {
        static const std::string static_case_name = (
            std::string("cublas")
            + ((flags & A_row_major_flag) ? ".Arow" : ".Acol")
            + ((flags & B_row_major_flag) ? ".Brow" : ".Bcol")
            + ((flags & C_row_major_flag) ? ".Crow" : ".Ccol")
        );

        return GemmCaseT<Ctype, ABtype>{
          CudaArch::Sm80,
          "sporkbench_builtin_cases.cu",
          static_case_name.c_str(),
          run,
          flags,
          1, INT32_MAX,  // L
          1, INT32_MAX,  // M
          1, INT32_MAX,  // N
          1, 1,  // K_split: set to 1, so we don't sweep this parameter.
          1, INT32_MAX,  // K_cluster
        };
    }
};

void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const float* A, const float* B, float* C)
{
    GemmEx<A_row_major_flag, float, float, float>::run(cublasH, size, A, B, C);
}

void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const __half* A, const __half* B, float* C)
{
    GemmEx<A_row_major_flag, float, __half, float>::run(cublasH, size, A, B, C);
}

void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const __half* A, const __half* B, __half* C)
{
    GemmEx<A_row_major_flag, __half, __half, __half>::run(cublasH, size, A, B, C);
}

void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const __nv_bfloat16* A, const __nv_bfloat16* B, float* C)
{
    GemmEx<A_row_major_flag, float, exo_bf16, float>::run(cublasH, size, A, B, C);
}

void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const exo_e4m3* A, const exo_e4m3* B, float* C)
{
    GemmEx<A_row_major_flag, float, exo_e4m3, float>::run(cublasH, size, A, B, C);
}

void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const exo_e5m2* A, const exo_e5m2* B, float* C)
{
    GemmEx<A_row_major_flag, float, exo_e5m2, float>::run(cublasH, size, A, B, C);
}

static_assert(std::variant_size_v<GemmCaseUnion> == 6, "Add more cublas cases");

void run_cublas_gemv(cublasHandle_t cublasH, GemvSize size, const float* A, const float* x, float* y)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    assert(cublasH);
    CUBLAS_CHECK(cublasSgemv(
            cublasH, CUBLAS_OP_T,
            size.K, size.M, &alpha,
            A, size.K,
            x, 1,
            &beta,
            y, 1));
}

static void run_pldi_Sm80_edited_exo_gemm(cublasHandle_t, GemmSize size, const __half* A, const __half* B, float* C)
{
    void* ctxt = nullptr;
    starter_ring_smem_gemm_2(ctxt, size.L, size.M, size.N, size.K_cluster, C, A, B);
}

template <typename Ctype, typename ABtype>
std::vector<GemmCaseT<Ctype, ABtype>> make_builtin_cases_gemm(const GemmCaseT<Ctype, ABtype>&)
{
    static_assert(all_row_major_flags == 7, "Update GemmEx case list");
    std::vector<GemmCaseT<Ctype, ABtype>> result {
      GemmEx<0, Ctype, ABtype, Ctype>::make_case(),
      GemmEx<1, Ctype, ABtype, Ctype>::make_case(),
      GemmEx<2, Ctype, ABtype, Ctype>::make_case(),
      GemmEx<3, Ctype, ABtype, Ctype>::make_case(),
      GemmEx<4, Ctype, ABtype, Ctype>::make_case(),
      GemmEx<5, Ctype, ABtype, Ctype>::make_case(),
      GemmEx<6, Ctype, ABtype, Ctype>::make_case(),
      GemmEx<7, Ctype, ABtype, Ctype>::make_case(),
    };

    if constexpr (std::is_same_v<ABtype, float> || std::is_same_v<ABtype, __half>) {
        result.push_back(make_cutlass_Sm80_GemmCase(Ctype{}, ABtype{}));
    }

    if constexpr (std::is_same_v<GemmCaseT<Ctype, ABtype>, GemmCase_f32_f16>) {
        result.push_back(GemmCase_f32_f16 {
            CudaArch::Sm80,
            "exocc_Sm80_edited.cuh",
            "pldi_Sm80_edited_exo",
            run_pldi_Sm80_edited_exo_gemm,
            A_row_major_flag | C_row_major_flag,
            1, 2147483647,  // L
            128, 2147483647,  // M
            128, 2147483647,  // N
            1, 1,  // K_split
            128, 2147483647,  // K_cluster
          }
      );
    }

    return result;
}

const std::vector<GemmCase_f32_f32>& get_builtin_cases(const GemmCase_f32_f32& arg)
{
    const static std::vector<GemmCase_f32_f32> saved = make_builtin_cases_gemm(arg);
    return saved;
}

const std::vector<GemmCase_f32_f16>& get_builtin_cases(const GemmCase_f32_f16& arg)
{
    const static std::vector<GemmCase_f32_f16> saved = make_builtin_cases_gemm(arg);
    return saved;
}

const std::vector<GemmCase_f16_f16>& get_builtin_cases(const GemmCase_f16_f16& arg)
{
    const static std::vector<GemmCase_f16_f16> saved = make_builtin_cases_gemm(arg);
    return saved;
}

const std::vector<GemmCase_f32_bf16>& get_builtin_cases(const GemmCase_f32_bf16& arg)
{
    const static std::vector<GemmCase_f32_bf16> saved = make_builtin_cases_gemm(arg);
    return saved;
}

const std::vector<GemmCase_f32_e4m3>& get_builtin_cases(const GemmCase_f32_e4m3& arg)
{
    const static std::vector<GemmCase_f32_e4m3> saved = make_builtin_cases_gemm(arg);
    return saved;
}

const std::vector<GemmCase_f32_e5m2>& get_builtin_cases(const GemmCase_f32_e5m2& arg)
{
    const static std::vector<GemmCase_f32_e5m2> saved = make_builtin_cases_gemm(arg);
    return saved;
}

static const GemvCase builtin_gemv_cases[] = {
  GemvCase{
    CudaArch::Sm80,
    "sporkbench_builtin_cases.cu",
    "cublas_gemv",
    run_cublas_gemv,
    1, INT32_MAX,  // M
    1, INT32_MAX,  // K
  },
};

const std::vector<GemvCase>& get_builtin_cases(const GemvCase&)
{
    constexpr size_t N = sizeof(builtin_gemv_cases) / sizeof(builtin_gemv_cases[0]);
    static const std::vector<GemvCase> result(&builtin_gemv_cases[0], &builtin_gemv_cases[N]);
    return result;
}

template <typename T_type, typename L_type, int Hdim, bool Causal>
std::vector<AttnFwdCaseT<T_type, L_type, Hdim, Causal>>
make_builtin_cases_attn_fwd()
{
    auto kittens_run = [] (
            AttnFwdSize size, exo_bf16* O, float* l_vec, const exo_bf16* Q, const exo_bf16* K, const exo_bf16* V)
    {
        cudaStream_t exo_cudaStream{};
        kittens_mha_h100::attention_forward(
                size.Batch, size.KV_Heads, size.Groups, size.SeqLen, size.Hdim, Causal,
                O, l_vec, const_cast<exo_bf16*>(Q), const_cast<exo_bf16*>(K), const_cast<exo_bf16*>(V),
                exo_cudaStream);
    };

    AttnFwdCaseT<T_type, L_type, Hdim, Causal> kittens_case{};
    kittens_case.cuda_arch = CudaArch::Sm90a;
    kittens_case.json_name = "sporkbench_builtin_cases.cu";
    kittens_case.proc_name = "kittens_mha_h100_attention_forward";
    kittens_case.run_function = kittens_run;
    kittens_case.Batch_divisor = 1;
    kittens_case.Batch_max = INT32_MAX;
    kittens_case.KV_Heads_divisor = 1;
    kittens_case.KV_Heads_max = INT32_MAX;
    kittens_case.Groups_divisor = 1;
    kittens_case.Groups_max = INT32_MAX;
    kittens_case.SeqLen_divisor = 16;
    kittens_case.SeqLen_max = INT32_MAX;
    std::vector<AttnFwdCaseT<T_type, L_type, Hdim, Causal>> cases{kittens_case};
    return cases;
}

const std::vector<AttnFwdCase_bf16_f32_64>& get_builtin_cases(const AttnFwdCase_bf16_f32_64&)
{
    const static auto saved = make_builtin_cases_attn_fwd<exo_bf16, float, 64, false>();
    return saved;
}

const std::vector<AttnFwdCase_bf16_f32_128>& get_builtin_cases(const AttnFwdCase_bf16_f32_128&)
{
    const static auto saved = make_builtin_cases_attn_fwd<exo_bf16, float, 128, false>();
    return saved;
}

const std::vector<AttnFwdCase_bf16_f32_64_causal>& get_builtin_cases(const AttnFwdCase_bf16_f32_64_causal&)
{
    const static auto saved = make_builtin_cases_attn_fwd<exo_bf16, float, 64, true>();
    return saved;
}

const std::vector<AttnFwdCase_bf16_f32_128_causal>& get_builtin_cases(const AttnFwdCase_bf16_f32_128_causal&)
{
    const static auto saved = make_builtin_cases_attn_fwd<exo_bf16, float, 128, true>();
    return saved;
}

}  // end namespace
