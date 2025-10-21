#include "sporkbench_cutlass_Sm80.hpp"

#include <stdio.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"


namespace sporkbench {

// https://github.com/NVIDIA/cutlass/blob/main/examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm.cu

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = float;                        // <- data type of elements in input matrix A
using ElementInputB = float;                        // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 16>;  // <- threadblock tile M = 128, N = 128, K = 16
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 16>;  // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 4;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;

static void run_cutlass_Sm80_gemm(cublasHandle_t, GemmSize size, const float* A, const float* B, float* C)
{
    const int M = size.M;
    const int N = size.N;
    const int K = size.K_split * size.K_cluster;
    cudaStream_t stream{};
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    const cutlass::layout::ColumnMajor C_layout{M};
    cutlass::TensorRef<const ElementInputA, LayoutInputA> A_ref {A, cutlass::layout::RowMajor{K}};
    cutlass::TensorRef<const ElementInputB, LayoutInputB> B_ref {B, cutlass::layout::ColumnMajor{K}};
    cutlass::TensorRef<const ElementOutput, LayoutOutput> C_ref {C, C_layout};
    cutlass::TensorRef<ElementOutput, LayoutOutput> D_ref {C, C_layout};

    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);
    int split_k_slices = size.K_split;
    typename Gemm::Arguments arguments{problem_size, A_ref, B_ref, C_ref, D_ref, {alpha, beta}, split_k_slices};
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    Gemm gemm_op;

    // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (cutlass::Status::kSuccess != status) {
        fprintf(stderr, "%s:%i gemm_op.can_implement\n", __FILE__, __LINE__);
        return;
    }

    void* workspace = 0;
    if (workspace_size > 0) {
        cudaMallocAsync(&workspace, workspace_size, stream);
        if (!workspace) {
            fprintf(stderr, "%s:%i memory allocation failed\n", __FILE__, __LINE__);
            return;
        }
    }

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace);
    if (cutlass::Status::kSuccess != status) {
        fprintf(stderr, "%s:%i gemm_op.initialize\n", __FILE__, __LINE__);
        return;
    }

    gemm_op(stream);

    if (workspace) {
        cudaFreeAsync(workspace, stream);
    }
}

extern const GemmCase cutlass_Sm80_GemmCase = {
    CudaArch::Sm80,
    "sporkbench_cutlass_Sm80.cu",
    "cutlass_Sm80_gemm",
    run_cutlass_Sm80_gemm,
    A_row_major_flag,
    1, 1,  // L
    16, INT32_MAX,  // M
    16, INT32_MAX,  // N
    1, 1,  // K_split
    16, INT32_MAX,  // K_cluster
};

}  // end namespace
