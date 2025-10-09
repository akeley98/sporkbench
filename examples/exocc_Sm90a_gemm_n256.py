from Sm90_gemm_impl import make_Sm90_gemm

cases = []

xgemm_Sm90_n256 = make_Sm90_gemm(256, 2, 1)
cases.append({
    "algorithm": "gemm",
    "proc": "xgemm_Sm90_wgmma_n256",
    "args": ["L", "M", "N", "K_split", "K_cluster", "A", "B", "C"],
    "K_split_max": 1,
})

import json
json.dump(cases, open(__file__ + ".json", "w"))
