from Sm90a_gemm_pre_config import config

config.smem_M = 256
config.smem_N = 192
config.tma_to_gmem = False
config.enable_split_k = False

from Sm90a_gen_gemm import *
import json
json.dump(cases, open(__file__ + ".json", "w"))
