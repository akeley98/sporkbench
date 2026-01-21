#include "sporkbench_cutlass_Sm100a_impl.cuh"

namespace sporkbench {

GemmCase_f16_f16 make_cutlass_Sm100a_GemmCase(__half, __half)
{
    // TODO implement.
    return {
        CudaArch::Sm100a,
        "sporkbench_cutlass_Sm100a_impl.cuh",
        "cutlass_Sm100a_gemm",
        [] (cublasHandle_t, GemmSize, const __half*, const __half*, __half*) {},
        A_row_major_flag | C_row_major_flag,
        1, 0,  // L
        16, 0,  // M
        16, 0,  // N
        1, 0,  // K_split
        16, 0,  // K_cluster
    };
    // return cutlass_Sm100a_gemm::make_GemmCase<cutlass::half_t, cutlass::half_t, cutlass::half_t, __half, __half>();
}

}  // end namespace
