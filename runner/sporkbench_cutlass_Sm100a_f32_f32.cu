#include "sporkbench_cutlass_Sm100a_impl.cuh"

namespace sporkbench {

GemmCase_f32_f32 make_cutlass_Sm100a_GemmCase(float, float)
{
    // TODO implement.
    return {
        CudaArch::Sm100a,
        "sporkbench_cutlass_Sm100a_impl.cuh",
        "cutlass_Sm100a_gemm",
        [] (cublasHandle_t, GemmSize, const float*, const float*, float*) {},
        A_row_major_flag | C_row_major_flag,
        1, 0,  // L
        16, 0,  // M
        16, 0,  // N
        1, 0,  // K_split
        16, 0,  // K_cluster
    };
    // return cutlass_Sm100a_gemm::make_GemmCase<float, float, float, float, float>();
}

}  // end namespace
