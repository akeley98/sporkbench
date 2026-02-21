from template.Sm90a_gemm_pre_config import Sm90aGemmConfig
from template.make_Sm90a_gemm import make_Sm90a_gemm

cases = []

def helper(M, N, ncta_M, ncta_N, tma_to_gmem, ring=None):
    config = Sm90aGemmConfig()
    config.smem_M = M
    config.smem_N = N
    config.tma_to_gmem = tma_to_gmem
    config.enable_split_k = False
    config.ping_pong = True
    if ring is not None:
        config.RING = ring
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

m128n128_m1n1_tma = helper(128, 128, 1, 1, tma_to_gmem=True)
m128n128_m1n2_tma = helper(128, 128, 1, 2, tma_to_gmem=True)
m256n96_m1n1_tma = helper(256, 96, 1, 1, tma_to_gmem=True, ring=3)
m256n96_m1n2_tma = helper(256, 96, 1, 2, tma_to_gmem=True, ring=3)

m128n128_m1n1 = helper(128, 128, 1, 1, tma_to_gmem=False)
m128n128_m1n2 = helper(128, 128, 1, 2, tma_to_gmem=False)
m256n96_m1n1 = helper(256, 96, 1, 1, tma_to_gmem=False)
m256n96_m1n2 = helper(256, 96, 1, 2, tma_to_gmem=False)

import json
json.dump(cases, open(__file__ + ".json", "w"))

