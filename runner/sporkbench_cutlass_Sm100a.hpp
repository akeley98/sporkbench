#include "sporkbench_cases.hpp"

namespace sporkbench {

GemmCase_f32_f32 make_cutlass_Sm100a_GemmCase(float, float);
GemmCase_f32_f16 make_cutlass_Sm100a_GemmCase(float, __half);
GemmCase_f16_f16 make_cutlass_Sm100a_GemmCase(__half, __half);

}  // end namespace
