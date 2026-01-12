#include "sporkbench_cutlass_Sm80_impl.cuh"

namespace sporkbench {

GemmCase_f16_f16 make_cutlass_Sm80_GemmCase(__half, __half)
{
    return cutlass_gemm::make_GemmCase<cutlass::half_t, cutlass::half_t, cutlass::half_t, __half, __half>();
}

}  // end namespace
