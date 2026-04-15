from exo.platforms.Sm90.tk_gemm_util import *

from dataclasses import replace  # Overrides Exo replace

cases = []

basic_config = GemmConfig(
    ncta_M=1,
    ncta_N=1,
    cta_M=128,
    cta_N=256,
    ring_depth=4,
    swizzle=128,
    A_type="bf16",
    A_major="row",
    B_type="bf16",
    B_major="col",
    C_type="f32",
    C_major="row",
    enable_split_k=False,
    ping_pong=False,
)

m1n1 = handwrite_gemm(replace(basic_config, ncta_M=1, ncta_N=1), cases)
m1n2 = handwrite_gemm(replace(basic_config, ncta_M=1, ncta_N=2), cases)
m2n1 = handwrite_gemm(replace(basic_config, ncta_M=2, ncta_N=1), cases)
m2n2 = handwrite_gemm(replace(basic_config, ncta_M=2, ncta_N=2), cases)


import json
json.dump(cases, open(__file__ + ".json", "w"))
