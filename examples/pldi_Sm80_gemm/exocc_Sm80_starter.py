from __future__ import annotations

from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *


@proc
def starter_gemm(M: size, N: size, K: size, C: f32[M, N] @ CudaGmemLinear, A: f16[M, K] @ CudaGmemLinear, B: f16[N, K] @ CudaGmemLinear):
    assert M % 128 == 0
    assert N % 128 == 0
    assert K % 8 == 0
    with CudaDeviceFunction(blockDim=128, blocks_per_sm=8):
        for m_task in cuda_tasks(0, (M + 127) / 128):
            for n_task in cuda_tasks(0, (N + 127) / 128):
                D_rmem: f32[2, 2, 4, 8, 16, 8] @ Sm80_RmemMatrixD_m16n8
                for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                    for nw in cuda_threads(0, 2, unit=cuda_warp):
                        for ms in seq(0, 4):
                            for ns in seq(0, 8):
                                Sm80_mma_m16n8_zero(D_rmem[mw, nw, ms, ns, :, :], D=f32)
                for ks in seq(0, K / 8):
                    for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                        for nw in cuda_threads(0, 2, unit=cuda_warp):
                            A_rmem: f16[8, 4, 8, 2] @ CudaRmemPacked32
                            B_rmem: f16[8, 4, 8, 2] @ CudaRmemPacked32
                            A_tile = A[m_task * 128 + mw * 64 : m_task * 128 + mw * 64 + 64, ks * 8 : ks * 8 + 8]
                            B_tile = B[n_task * 128 + nw * 64 : n_task * 128 + nw * 64 + 64, ks * 8 : ks * 8 + 8]
                            for mt in cuda_threads(0, 8, unit=4 * cuda_thread):
                                for kt in cuda_threads(0, 4, unit=cuda_thread):
                                    for ms in seq(0, 8):
                                        cuda_packed32_load(
                                            A_rmem[mt, kt, ms, 0:2],
                                            A_tile[ms * 8 + mt, kt * 2 : kt * 2 + 2],
                                            pack=2, dst=f16, src=f16,
                                        )
                                        cuda_packed32_load(
                                            B_rmem[mt, kt, ms, 0:2],
                                            B_tile[ms * 8 + mt, kt * 2 : kt * 2 + 2],
                                            pack=2, dst=f16, src=f16,
                                        )
                            for ms in seq(0, 4):
                                for ns in seq(0, 8):
                                    Sm80_mma_m16n8(
                                        D_rmem[mw, nw, ms, ns, :, :],
                                        A_rmem[:, :, ms * 2 : ms * 2 + 2, :],
                                        B_rmem[:, :, ns, :],
                                        D=f32,
                                        A=f16,
                                        B=f16,
                                        K_pack=2,
                                    )
                for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                    for nw in cuda_threads(0, 2, unit=cuda_warp):
                        C_tile = C[
                            m_task * 128 + mw * 64 : m_task * 128 + mw * 64 + 64,
                            n_task * 128 + nw * 64 : n_task * 128 + nw * 64 + 64,
                        ]
                        for ms in seq(0, 4):
                            for ns in seq(0, 8):
                                # TODO this is legacy.
                                Sm80_mma_store_d_row_major_tf32(
                                    C_tile[ms * 16 : ms * 16 + 16, ns * 8 : ns * 8 + 8],
                                    D_rmem[mw, nw, ms, ns, :, :],
                                )


cases = [{
    "algorithm": "gemm",
    "proc": "starter_gemm",
    "args": ["M", "N", "K", "C", "A", "B"],
    "A_major": "row",
    "B_major": "col",
    "C_major": "row",
    "M_divisor": 128,
    "N_divisor": 128,
    "K_divisor": 8,
    "A_type": "f16",
    "B_type": "f16",
    "C_type": "f32",
}]

import json
json.dump(cases, open(__file__ + ".json", "w"))
