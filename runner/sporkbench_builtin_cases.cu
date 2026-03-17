#include "sporkbench_cases.hpp"

#include <cassert>
#include <cublas_v2.h>
#include <stdexcept>
#include <stdio.h>
#include <type_traits>

#include "sporkbench_cutlass_Sm80.hpp"

#include "pldi_Sm80_edited/exocc_Sm80_edited.h"

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
                &beta, C, M));
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

template <typename Ctype, typename ABtype, typename ComputeType>
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
        return CUDA_R_8F_E4M3;
    }

    static cudaDataType_t get_type_code(exo_e5m2)
    {
        return CUDA_R_8F_E5M2;
    }

    static cudaDataType_t get_type_code(exo_e8m0)
    {
        throw std::runtime_error("TODO GemmEx::get_type_code(exo_e8m0)");
    }

    static void run(cublasHandle_t cublasH, GemmSize size, const ABtype* A, const ABtype* B, Ctype* C)
    {
        assert(cublasH);
        cublasOperation_t transa = CUBLAS_OP_T;
        cublasOperation_t transb = CUBLAS_OP_N;
        const ComputeType alpha = 1.0f;
        const ComputeType beta = 0.0f;
        const int M = int(size.M);
        const int N = int(size.N);
        const int K = int(size.K_split * size.K_cluster);
        if (size.L == 1) {
            CUBLAS_CHECK(cublasGemmEx(
                    cublasH, transa, transb,
                    M, N, K, &alpha,
                    A, get_type_code(ABtype{}), K,
                    B, get_type_code(ABtype{}), K,
                    &beta, C, get_type_code(Ctype{}), M,
                    get_compute_code(ComputeType{}), CUBLAS_GEMM_DEFAULT));
        }
        else {
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                    cublasH, transa, transb,
                    M, N, K, &alpha,
                    A, get_type_code(ABtype{}), K, M * K,
                    B, get_type_code(ABtype{}), K, N * K,
                    &beta, C, get_type_code(Ctype{}), M, M * N, size.L,
                    get_compute_code(ComputeType{}), CUBLAS_GEMM_DEFAULT));
        }
    }
};

void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const __half* A, const __half* B, float* C)
{
    GemmEx<float, __half, float>::run(cublasH, size, A, B, C);
}

void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const __half* A, const __half* B, __half* C)
{
    GemmEx<__half, __half, __half>::run(cublasH, size, A, B, C);
}

void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const __nv_bfloat16* A, const __nv_bfloat16* B, float* C)
{
    GemmEx<float, exo_bf16, float>::run(cublasH, size, A, B, C);
}

void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const exo_e4m3* A, const exo_e4m3* B, float* C)
{
    GemmEx<float, exo_e4m3, float>::run(cublasH, size, A, B, C);
}

void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const exo_e5m2* A, const exo_e5m2* B, float* C)
{
    GemmEx<float, exo_e5m2, float>::run(cublasH, size, A, B, C);
}

void run_cublas_gemm(cublasHandle_t cublasH, GemmSize size, const exo_e8m0* A, const exo_e8m0* B, float* C)
{
    GemmEx<float, exo_e8m0, float>::run(cublasH, size, A, B, C);
}

static_assert(std::variant_size_v<GemmCaseUnion> == 7, "Add more cublas cases");

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
    std::vector<GemmCaseT<Ctype, ABtype>> result {
      GemmCaseT<Ctype, ABtype>{
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

const std::vector<GemmCase_f32_e8m0>& get_builtin_cases(const GemmCase_f32_e8m0& arg)
{
    const static std::vector<GemmCase_f32_e8m0> saved = make_builtin_cases_gemm(arg);
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

}  // end namespace
