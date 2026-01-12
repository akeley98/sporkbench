from __future__ import annotations

from exo import proc
from exo.platforms.cuda import *
from exo.platforms.Sm90 import Sm90_SmemSwizzled
from exo.stdlib.scheduling import *

def mkproc_naive_gemm(typA, typB, typC):
    @proc
    def p(
        M: size,
        N: size,
        K: size,
        d_A: f16[M, K] @ CudaGmemLinear,
        d_B: f16[N, K] @ CudaGmemLinear,
        d_C: f32[N, M] @ CudaGmemLinear,
    ):
        assert M % 128 == 0
        assert N % 128 == 0
        assert K % 32 == 0

        with CudaDeviceFunction(blockDim=128, blocks_per_sm=3):
            for m_cta in cuda_tasks(0, M / 128):
                for n_cta in cuda_tasks(0, N / 128):
                    s_A: f16[128, 32] @ Sm90_SmemSwizzled(128)
                    s_B: f16[128, 32] @ CudaSmemLinear
                    r_C: f32[128, 128] @ CudaRmem

                    for m in cuda_threads(0, 128):
                        for n in seq(0, 128):
                            r_C[m, n] = 0

                    for ks in seq(0, K / 32):
                        for ms in seq(0, 32):
                            for mt in cuda_threads(0, 4, unit=32 * cuda_thread):
                                for k in cuda_threads(0, 32):
                                    s_A[ms * 4 + mt, k] = d_A[
                                        m_cta * 128 + ms * 4 + mt, ks * 32 + k
                                    ]
                                    s_B[ms * 4 + mt, k] = d_B[
                                        n_cta * 128 + ms * 4 + mt, ks * 32 + k
                                    ]
                        Fence(cuda_in_order, cuda_in_order)

                        for m in cuda_threads(0, 128):
                            for n in seq(0, 128):
                                for k in seq(0, 32):
                                    a_f32: f32 @ CudaRmem
                                    b_f32: f32 @ CudaRmem
                                    a_f32 = s_A[m, k]
                                    b_f32 = s_B[n, k]
                                    r_C[m, n] += a_f32 * b_f32
                        Fence(cuda_in_order, cuda_in_order)

                    for m in cuda_threads(0, 128):
                        for n in seq(0, 128):
                            d_C[n_cta * 128 + n, m_cta * 128 + m] = r_C[m, n]

    p = set_precision(p, "d_A", typA)
    p = set_precision(p, "s_A", typA)
    p = set_precision(p, "d_B", typB)
    p = set_precision(p, "s_B", typB)
    p = set_precision(p, "d_C", typC)  # Leave r_C as f32

    p = rename(p, f"naive_gemm_{typC}_{typA}_{typB}")

    return p

cases = []

def add_case_helper(p: exo.Procedure, typA, typB, typC):
    cases.append({
        "algorithm": "gemm",
        "proc": p.name(),
        "args": ["M", "N", "K", "A", "B", "C"],
        "A_major": "row", "B_major": "col", "C_major": "col",
        "M_divisor": 128, "N_divisor": 128, "K_divisor": 32,
        "A_type": typA, "B_type": typB, "C_type": typC,
        # Disable this gemm at a certain point to avoid wasting too much time.
        "M_max": 2048,
        "N_max": 2048,
        "K_max": 2048,
    })

naive_gemm_f32_f16 = mkproc_naive_gemm(typA="f16", typB="f16", typC="f32")
add_case_helper(naive_gemm_f32_f16, typA="f16", typB="f16", typC="f32")

naive_gemm_f16_f16 = mkproc_naive_gemm(typA="f16", typB="f16", typC="f16")
add_case_helper(naive_gemm_f16_f16, typA="f16", typB="f16", typC="f16")

naive_gemm_f32_bf16 = mkproc_naive_gemm(typA="bf16", typB="bf16", typC="f32")
# add_case_helper(naive_gemm_f32_bf16, typA="bf16", typB="bf16", typC="f32")

naive_gemm_bf16_bf16 = mkproc_naive_gemm(typA="bf16", typB="bf16", typC="bf16")
# add_case_helper(naive_gemm_bf16_bf16, typA="bf16", typB="bf16", typC="bf16")

naive_gemm_f32_f16.sync_check(M=256, N=128, K=64)

import json
json.dump(cases, open(__file__ + ".json", "w"))

