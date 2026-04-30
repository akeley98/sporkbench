from __future__ import annotations

from exo.platforms.Sm90.tk_gemm_util import handwrite_gemm, GemmConfig

from dataclasses import replace

base_config = GemmConfig(
    A_type="f16",
    B_type="f16",
    C_type="f16",
    A_major="row",
    B_major="row",
    C_major="row",
    cta_M=256,
    cta_N=192,
    enable_split_k=False,
)

cases = []

gemm_m1n1 = handwrite_gemm(replace(base_config, ncta_M=1, ncta_N=1), cases)
gemm_m2n1 = handwrite_gemm(replace(base_config, ncta_M=2, ncta_N=1), cases)
gemm_m1n2 = handwrite_gemm(replace(base_config, ncta_M=1, ncta_N=2), cases)

import json
json.dump(cases, open(__file__ + ".json", "w"))
