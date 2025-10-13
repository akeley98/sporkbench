from Sm80_gemm_pre_config import config
config.smem_M = 128
config.smem_N = 256
config.smem_K = 16
config.warp_M = 64
config.warp_N = 64
config.enable_split_k = False
config.blocks_per_sm = 1

from Sm80_gen_gemm import *
import json
json.dump(cases, open(__file__ + ".json", "w"))
