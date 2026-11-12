# Temporary: a few handwritten ping-pong gemm kernels (exo tk_gemm_util
# handwrite_row_col_ping_pong_gemm), one per cluster multicast case.
# Each warpgroup computes a (cta_M / 2, cta_N) tile.

from __future__ import annotations
from exo.platforms.Sm90.tk_gemm_util import handwrite_gemm, GemmConfig, f32, bf16
from dataclasses import replace

base_config = GemmConfig(
    A_type=bf16,
    B_type=bf16,
    C_type=f32,
    D_type=f32,
    A_major="row",
    B_major="col",
    C_major="row",
    cta_M=256,
    cta_N=128,
    ring_depth=4,
    enable_split_k=False,
    ping_pong=True,
)

cases = []

gemm_m1n1 = handwrite_gemm(replace(base_config, ncta_M=1, ncta_N=1), cases)
gemm_m2n1 = handwrite_gemm(replace(base_config, ncta_M=2, ncta_N=1), cases)
gemm_m1n2 = handwrite_gemm(replace(base_config, ncta_M=1, ncta_N=2), cases)
gemm_m2n2 = handwrite_gemm(replace(base_config, ncta_M=2, ncta_N=2), cases)

import json

json.dump(cases, open(__file__ + ".json", "w"))
