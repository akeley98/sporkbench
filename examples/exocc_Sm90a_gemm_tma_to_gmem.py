from Sm90_gemm_impl import make_Sm90_gemm

cases = []

xgemm_Sm90_wgmma_n256_tma_to_gmem = make_Sm90_gemm(256, 2, 1, tma_to_gmem=True)
cases.append({
    "algorithm": "gemm",
    "proc": "xgemm_Sm90_wgmma_n256_tma_to_gmem",
    "args": ["L", "M", "N", "K_split", "K_cluster", "A", "B", "C"],
    "K_split_max": 1,
    "A_major": "row", "B_major": "col", "C_major": "col",
})

xgemm_Sm90_wgmma_n256_split_k = make_Sm90_gemm(256, 2, 1, tma_to_gmem=True, enable_split_k=True)
cases.append({
    "algorithm": "gemm",
    "proc": "xgemm_Sm90_wgmma_n256_split_k",
    "args": ["L", "M", "N", "K_split", "K_cluster", "A", "B", "C"],
    "A_major": "row", "B_major": "col", "C_major": "col",
})

import json
json.dump(cases, open(__file__ + ".json", "w"))
