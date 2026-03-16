from template_symlink.Sm90a_gemm_pre_config import Sm90aGemmConfig
from template_symlink.make_tk_Sm90a_gemm import make_Sm90a_gemm

cases = []

import os
thisdir = os.path.split(__file__)[0]

config = Sm90aGemmConfig()
config.smem_M = 128
config.smem_N = 256
config.tma_to_gmem = False
config.enable_split_k = False

gemm_m1n1 = make_Sm90a_gemm(config, 1, 1, cases)
gemm_m1n2 = make_Sm90a_gemm(config, 1, 2, cases)
gemm_m2n1 = make_Sm90a_gemm(config, 2, 1, cases)

cases = []  # XXX

import json
json.dump(cases, open(__file__ + ".json", "w"))
