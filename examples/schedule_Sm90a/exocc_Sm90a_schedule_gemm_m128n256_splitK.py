from __future__ import annotations

from template_symlink.Sm90a_gemm_pre_config import Sm90aGemmConfig
from template_symlink.schedule_Sm90a_gemm import schedule_Sm90a_gemm

cases = []

from template_symlink.make_Sm90a_gemm import make_Sm90a_gemm
import os
thisdir = os.path.split(__file__)[0]

config = Sm90aGemmConfig()
config.smem_M = 128
config.smem_N = 256
config.tma_to_gmem = True
config.enable_split_k = True

gemm_m1n1 = schedule_Sm90a_gemm(config, 1, 1, cases)
gemm_m1n2 = schedule_Sm90a_gemm(config, 1, 2, cases)
gemm_m2n1 = schedule_Sm90a_gemm(config, 2, 1, cases)

import json
json.dump(cases, open(__file__ + ".json", "w"))

