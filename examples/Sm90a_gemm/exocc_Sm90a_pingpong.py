from template.Sm90a_gemm_pre_config import Sm90aGemmConfig
from template.make_Sm90a_gemm import make_Sm90a_gemm

cases = []

def helper(M, N, ncta_M, ncta_N):
    config = Sm90aGemmConfig()
    config.smem_M = M
    config.smem_N = N
    config.tma_to_gmem = False
    config.enable_split_k = False
    config.ping_pong = True
    p = make_Sm90a_gemm(config, ncta_M, ncta_N)
    cases.append({
        "algorithm": "gemm",
        "proc": p.name(),
        "args": ["L", "M", "N", "K_split", "K_cluster", "A", "B", "C"],
        "K_split_max": 1,
        "A_major": "row", "B_major": "col", "C_major": "col",
        "A_type": "f32", "B_type": "f32", "C_type": "f32",
    })
    return p

m128n128_m1n1 = helper(128, 128, 1, 1)
m128n128_m1n2 = helper(128, 128, 1, 2)
m256n96_m1n1 = helper(256, 96, 1, 1)
m256n96_m1n2 = helper(256, 96, 1, 2)

import json
json.dump(cases, open(__file__ + ".json", "w"))

