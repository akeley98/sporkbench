from __future__ import annotations
from exo import *
from exo.stdlib.scheduling import *
from exo.stdlib.stdlib import auto_stage_mem
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *


from template_symlink.Sm90a_gemm_pre_config import Sm90aGemmConfig
from template_symlink.schedule_Sm90a_gemm import schedule_Sm90a_gemm

cases = []

from template_symlink.make_Sm90a_gemm import make_Sm90a_gemm
import os
thisdir = os.path.split(__file__)[0]

config = Sm90aGemmConfig()
config.smem_M = 128
config.smem_N = 256
config.tma_to_gmem = False
config.enable_split_k = False
ncta_M, ncta_N = 2, 1

scheduled_gemm = schedule_Sm90a_gemm(config, ncta_M, ncta_N)
scheduled_gemm = rename(scheduled_gemm, "scheduled_gemm")
scheduled_gemm.sync_check(L=2, M=500, N=800, cluster_K=240, K_split=1)

def debug_dump_comparison():
    handwritten_gemm = make_Sm90a_gemm(config, ncta_M, ncta_N)
    handwritten_gemm = rename(handwritten_gemm, "handwritten_gemm")
    open(os.path.join(thisdir, "out_scheduled_gemm.py"), "w").write(str(scheduled_gemm))
    open(os.path.join(thisdir, "out_handwritten_gemm.py"), "w").write(str(handwritten_gemm))
debug_dump_comparison()

cases.append({
    "algorithm": "gemm",
    "proc": "scheduled_gemm",
    "args": ["L", "M", "N", "K_split", "K_cluster", "A", "B", "C"],
    "K_split_max": 1,
    "A_major": "row", "B_major": "col", "C_major": "col",
})

import json
json.dump(cases, open(__file__ + ".json", "w"))

