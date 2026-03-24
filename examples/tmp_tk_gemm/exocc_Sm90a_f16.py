from __future__ import annotations

from template_symlink.make_tk_Sm90a_gemm import make_Sm90a_generic_gemm

import time

from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.platforms.cuda_tk import *

from exo.scalars import e4m3, e5m2, bf16, f16, f32

from typing import List

cases = []

gemm_f32_f16_f16_m1n1 = make_Sm90a_generic_gemm(1, 1, f32, f16, f16, cases)
gemm_f32_f16_f16_m1n2 = make_Sm90a_generic_gemm(1, 2, f32, f16, f16, cases)
gemm_f32_f16_f16_m2n1 = make_Sm90a_generic_gemm(2, 1, f32, f16, f16, cases)
gemm_f32_f16_f16_m2n2 = make_Sm90a_generic_gemm(2, 2, f32, f16, f16, cases)

# bf16 and fp8 test, so file name is a misnomer...
gemm_f32_bf16_bf16_m1n1 = make_Sm90a_generic_gemm(1, 1, f32, bf16, bf16, cases)
# These don't work yet we have to fix cublas usage.
if False:
    gemm_f32_e4m3_e4m3_m1n1 = make_Sm90a_generic_gemm(1, 1, f32, e4m3, e4m3, cases)
    gemm_f32_e5m2_e5m2_m1n1 = make_Sm90a_generic_gemm(1, 1, f32, e5m2, e5m2, cases)
    gemm_f32_e5m2_e5m2_m2n1 = make_Sm90a_generic_gemm(2, 1, f32, e5m2, e5m2, cases)

import json
json.dump(cases, open(__file__ + ".json", "w"))
