#include "sporkbench_cutlass_Sm80_impl.cuh"

namespace sporkbench {

GemmCase_f32_f32 make_cutlass_Sm80_GemmCase(float, float)
{
    return cutlass_gemm::make_GemmCase<float, float, float, float, float>();
}

}  // end namespace
