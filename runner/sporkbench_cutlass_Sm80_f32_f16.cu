#include "sporkbench_cutlass_Sm80_impl.cuh"

namespace sporkbench {

GemmCase_f32_f16 make_cutlass_Sm80_GemmCase(float, __half)
{
    return cutlass_gemm::make_GemmCase<float, float, cutlass::half_t, float, __half>();
}

}  // end namespace
