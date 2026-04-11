from __future__ import annotations

from template_symlink.make_tk_Sm90a_gemm import make_Sm90a_generic_gemm_Brow

import time

from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.platforms.cuda_tk import *

from exo.scalars import e4m3, e5m2, e8m0, bf16, f16, f32

from typing import List

cases = []

# gemm_m1n1_f32_bf16_rmem_row = make_Sm90a_generic_gemm_Brow(1, 1, f32, bf16, bf16, f32, "rmem", cases)
# gemm_m2n1_f32_bf16_rmem_row = make_Sm90a_generic_gemm_Brow(2, 1, f32, bf16, bf16, f32, "rmem", cases)

gemm_m1n1_f32_f16 = make_Sm90a_generic_gemm_Brow(1, 1, f32, f16, f16, f32, "row", cases)
# gemm_m1n2_f32_f16 = make_Sm90a_generic_gemm_Brow(1, 2, f32, f16, f16, f32, "row", cases)
# gemm_m2n1_f32_f16 = make_Sm90a_generic_gemm_Brow(2, 1, f32, f16, f16, f32, "row", cases)
# gemm_m2n2_f32_f16 = make_Sm90a_generic_gemm_Brow(2, 2, f32, f16, f16, f32, "row", cases)

gemm_m1n1_f32_bf16 = make_Sm90a_generic_gemm_Brow(1, 1, f32, bf16, bf16, f32, "row", cases)
# gemm_m1n2_f32_bf16 = make_Sm90a_generic_gemm_Brow(1, 2, f32, bf16, bf16, f32, "row", cases)
# gemm_m2n1_f32_bf16 = make_Sm90a_generic_gemm_Brow(2, 1, f32, bf16, bf16, f32, "row", cases)
# gemm_m2n2_f32_bf16 = make_Sm90a_generic_gemm_Brow(2, 2, f32, bf16, bf16, f32, "row", cases)

gemm_m1n1_f16_f16 = make_Sm90a_generic_gemm_Brow(1, 1, f16, f16, f16, f16, "row", cases)
# gemm_m1n2_f16_f16 = make_Sm90a_generic_gemm_Brow(1, 2, f16, f16, f16, f16, "row", cases)
# gemm_m2n1_f16_f16 = make_Sm90a_generic_gemm_Brow(2, 1, f16, f16, f16, f16, "row", cases)
# gemm_m2n2_f16_f16 = make_Sm90a_generic_gemm_Brow(2, 2, f16, f16, f16, f16, "row", cases)


import json
json.dump(cases, open(__file__ + ".json", "w"))
