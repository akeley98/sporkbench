#include "sporkbench_cutlass_Sm100a_impl.cuh"

namespace sporkbench {

GemmCase_f32_f16 make_cutlass_Sm100a_GemmCase(float, __half)
{
    return cutlass_Sm100a_gemm::make_GemmCase<float, float, cutlass::half_t, float, __half>();
}

}  // end namespace
