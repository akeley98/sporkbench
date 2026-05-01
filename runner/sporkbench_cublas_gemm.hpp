#pragma once

#include "sporkbench_cases.hpp"

#include <cassert>
#include <cublas_v2.h>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <type_traits>
#include <utility>

namespace sporkbench {

#define SPORKBENCH_RUNNER_CUBLAS_CHECK(x) if (auto _cublas_status = x; _cublas_status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%i cublas status %i\n", __FILE__, __LINE__, (int)_cublas_status); }

template <int flags, typename Ctype, typename ABtype, typename ComputeType>
struct GemmEx
{
    static cublasComputeType_t get_compute_code(__half)
    {
        return CUBLAS_COMPUTE_16F;
    }

    static cublasComputeType_t get_compute_code(__nv_bfloat16) = delete;

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
            SPORKBENCH_RUNNER_CUBLAS_CHECK(cublasGemmEx(
                    cublasH, transa, transb,
                    M, N, K, &alpha,
                    A, get_type_code(ABtype{}), lda,
                    B, get_type_code(ABtype{}), ldb,
                    &beta, C, get_type_code(Ctype{}), M,
                    get_compute_code(ComputeType{}), CUBLAS_GEMM_DEFAULT));
        }
        else {
            SPORKBENCH_RUNNER_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
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

}  // end namespace
