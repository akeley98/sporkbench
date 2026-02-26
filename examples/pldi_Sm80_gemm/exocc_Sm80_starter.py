from __future__ import annotations

from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *


cases = []


def add_case(p):
    p.sync_check(L=1, M=256, N=256, K=320)
    cases.append({
        "algorithm": "gemm",
        "proc": p.name(),
        "args": ["L", "M", "N", "K", "C", "A", "B"],
        "A_major": "row",
        "B_major": "col",
        "C_major": "row",
        "M_divisor": 128,
        "N_divisor": 128,
        "K_divisor": 64,
        "A_type": "f16",
        "B_type": "f16",
        "C_type": "f32",
    })


@proc
def starter_no_smem_gemm(L: size, M: size, N: size, K: size, C: f32[L, M, N] @ CudaGmemLinear, A: f16[L, M, K] @ CudaGmemLinear, B: f16[L, N, K] @ CudaGmemLinear):
    assert M % 128 == 0
    assert N % 128 == 0
    assert K % 16 == 0
    with CudaDeviceFunction(blockDim=128, blocks_per_sm=3):
        for batch in cuda_tasks(0, L):
            for m_task in cuda_tasks(0, (M + 127) / 128):
                for n_task in cuda_tasks(0, (N + 127) / 128):
                    D_rmem: f32[2, 2, 4, 8, 16, 8] @ Sm80_RmemMatrixD_m16n8
                    for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                        for nw in cuda_threads(0, 2, unit=cuda_warp):
                            for ms in seq(0, 4):
                                for ns in seq(0, 8):
                                    Sm80_mma_m16n8_zero(D_rmem[mw, nw, ms, ns, :, :], D=f32)
                    for ks in seq(0, K / 16):
                        for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                            for nw in cuda_threads(0, 2, unit=cuda_warp):
                                A_rmem: f16[8, 4, 8, 2, 2] @ CudaRmemPacked32
                                B_rmem: f16[8, 4, 8, 2, 2] @ CudaRmemPacked32
                                A_tile = A[batch, m_task * 128 + mw * 64 : m_task * 128 + mw * 64 + 64, ks * 16 : ks * 16 + 16]
                                B_tile = B[batch, n_task * 128 + nw * 64 : n_task * 128 + nw * 64 + 64, ks * 16 : ks * 16 + 16]
                                for mt in cuda_threads(0, 8, unit=4 * cuda_thread):
                                    for kt in cuda_threads(0, 4, unit=cuda_thread):
                                        for ms in seq(0, 8):
                                            for kr in seq(0, 2):
                                                cuda_packed32_load(
                                                    A_rmem[mt, kt, ms, kr, 0:2],
                                                    A_tile[ms * 8 + mt, kr * 8 + kt * 2 : kr * 8 + kt * 2 + 2],
                                                    pack=2, dst=f16, src=f16,
                                                )
                                                cuda_packed32_load(
                                                    B_rmem[mt, kt, ms, kr, 0:2],
                                                    B_tile[ms * 8 + mt, kr * 8 + kt * 2 : kr * 8 + kt * 2 + 2],
                                                    pack=2, dst=f16, src=f16,
                                                )
                                for ms in seq(0, 4):
                                    for ns in seq(0, 8):
                                        Sm80_mma_m16n8(
                                            D_rmem[mw, nw, ms, ns, :, :],
                                            A_rmem[:, :, ms * 2 : ms * 2 + 2, :, :],
                                            B_rmem[:, :, ns, :, :],
                                            D=f32,
                                            A=f16,
                                            B=f16,
                                            K_pack=2,
                                        )
                    for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                        for nw in cuda_threads(0, 2, unit=cuda_warp):
                            C_tile = C[
                                batch,
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


add_case(starter_no_smem_gemm)


@proc
def starter_simple_smem_gemm(L: size, M: size, N: size, K: size, C: f32[L, M, N] @ CudaGmemLinear, A: f16[L, M, K] @ CudaGmemLinear, B: f16[L, N, K] @ CudaGmemLinear):
    assert M % 128 == 0
    assert N % 128 == 0
    assert K % 16 == 0
    with CudaDeviceFunction(blockDim=128, blocks_per_sm=3):
        for batch in cuda_tasks(0, L):
            for m_task in cuda_tasks(0, (M + 127) / 128):
                for n_task in cuda_tasks(0, (N + 127) / 128):
                    D_rmem: f32[2, 2, 4, 8, 16, 8] @ Sm80_RmemMatrixD_m16n8
                    for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                        for nw in cuda_threads(0, 2, unit=cuda_warp):
                            for ms in seq(0, 4):
                                for ns in seq(0, 8):
                                    Sm80_mma_m16n8_zero(D_rmem[mw, nw, ms, ns, :, :], D=f32)

                    A_smem: f16[128, 16] @ CudaSmemLinear
                    B_smem: f16[128, 16] @ CudaSmemLinear

                    for ks in seq(0, K / 16):
                        A_tile = A[batch, m_task * 128 : m_task * 128 + 128, ks * 16 : ks * 16 + 16]
                        B_tile = B[batch, n_task * 128 : n_task * 128 + 128, ks * 16 : ks * 16 + 16]

                        for mns in seq(0, 2):
                            for mnt in cuda_threads(0, 64, unit=2 * cuda_thread):
                                for kt in cuda_threads(0, 2, unit=cuda_thread):
                                    Sm80_cp_async_1d(
                                        A_smem[mns * 64 + mnt, kt * 8 : kt * 8 + 8],
                                        A_tile[mns * 64 + mnt, kt * 8 : kt * 8 + 8],
                                        size0=8, dst=f16, src=f16,
                                    )
                                    Sm80_cp_async_1d(
                                        B_smem[mns * 64 + mnt, kt * 8 : kt * 8 + 8],
                                        B_tile[mns * 64 + mnt, kt * 8 : kt * 8 + 8],
                                        size0=8, dst=f16, src=f16,
                                    )

                        Fence(Sm80_cp_async, cuda_in_order)

                        for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                            for nw in cuda_threads(0, 2, unit=cuda_warp):
                                A_rmem: f16[8, 4, 8, 2, 2] @ CudaRmemPacked32
                                B_rmem: f16[8, 4, 8, 2, 2] @ CudaRmemPacked32

                                for kr in seq(0, 2):
                                    Sm80_ldmatrix_f16(
                                        A_rmem[:, :, 0:4, kr:kr+1, 0:2],
                                        A_smem[mw * 64 + 0 : mw * 64 + 32, 8*kr:8*kr+8],
                                        nmat0=4, nmat1=1,
                                    )
                                    Sm80_ldmatrix_f16(
                                        A_rmem[:, :, 4:8, kr:kr+1, 0:2],
                                        A_smem[mw * 64 + 32 : mw * 64 + 64, 8*kr:8*kr+8],
                                        nmat0=4, nmat1=1,
                                    )
                                    Sm80_ldmatrix_f16(
                                        B_rmem[:, :, 0:4, kr:kr+1, 0:2],
                                        B_smem[nw * 64 + 0 : nw * 64 + 32, 8*kr:8*kr+8],
                                        nmat0=4, nmat1=1,
                                    )
                                    Sm80_ldmatrix_f16(
                                        B_rmem[:, :, 4:8, kr:kr+1, 0:2],
                                        B_smem[nw * 64 + 32 : nw * 64 + 64, 8*kr:8*kr+8],
                                        nmat0=4, nmat1=1,
                                    )

                                for ms in seq(0, 4):
                                    for ns in seq(0, 8):
                                        Sm80_mma_m16n8(
                                            D_rmem[mw, nw, ms, ns, :, :],
                                            A_rmem[:, :, ms * 2 : ms * 2 + 2, :, :],
                                            B_rmem[:, :, ns, :, :],
                                            D=f32,
                                            A=f16,
                                            B=f16,
                                            K_pack=2,
                                        )
                        Fence(cuda_in_order, cuda_in_order)

                    for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                        for nw in cuda_threads(0, 2, unit=cuda_warp):
                            C_tile = C[
                                batch,
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


add_case(starter_simple_smem_gemm)


@proc
def starter_double_smem_gemm(L: size, M: size, N: size, K: size, C: f32[L, M, N] @ CudaGmemLinear, A: f16[L, M, K] @ CudaGmemLinear, B: f16[L, N, K] @ CudaGmemLinear):
    assert M % 128 == 0
    assert N % 128 == 0
    assert K % 16 == 0
    with CudaDeviceFunction(blockDim=128, blocks_per_sm=3):
       for batch in cuda_tasks(0, L):
            for m_task in cuda_tasks(0, (M + 127) / 128):
                for n_task in cuda_tasks(0, (N + 127) / 128):
                    D_rmem: f32[2, 2, 4, 8, 16, 8] @ Sm80_RmemMatrixD_m16n8
                    for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                        for nw in cuda_threads(0, 2, unit=cuda_warp):
                            for ms in seq(0, 4):
                                for ns in seq(0, 8):
                                    Sm80_mma_m16n8_zero(D_rmem[mw, nw, ms, ns, :, :], D=f32)

                    A_smem: f16[2, 128, 16] @ CudaSmemLinear
                    B_smem: f16[2, 128, 16] @ CudaSmemLinear

                    for ks in seq(0, K / 16 + 1):
                        A_tile = A[batch, m_task * 128 : m_task * 128 + 128, ks * 16 : ks * 16 + 16]
                        B_tile = B[batch, n_task * 128 : n_task * 128 + 128, ks * 16 : ks * 16 + 16]

                        if ks < K / 16:
                            for mns in seq(0, 2):
                                for mnt in cuda_threads(0, 64, unit=2 * cuda_thread):
                                    for kt in cuda_threads(0, 2, unit=cuda_thread):
                                        Sm80_cp_async_1d(
                                            A_smem[ks % 2, mns * 64 + mnt, kt * 8 : kt * 8 + 8],
                                            A_tile[mns * 64 + mnt, kt * 8 : kt * 8 + 8],
                                            size0=8, dst=f16, src=f16,
                                        )
                                        Sm80_cp_async_1d(
                                            B_smem[ks % 2,  mns * 64 + mnt, kt * 8 : kt * 8 + 8],
                                            B_tile[mns * 64 + mnt, kt * 8 : kt * 8 + 8],
                                            size0=8, dst=f16, src=f16,
                                        )

                        if ks > 0:
                            for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                                for nw in cuda_threads(0, 2, unit=cuda_warp):
                                    A_rmem: f16[8, 4, 8, 2, 2] @ CudaRmemPacked32
                                    B_rmem: f16[8, 4, 8, 2, 2] @ CudaRmemPacked32

                                    for kr in seq(0, 2):
                                        Sm80_ldmatrix_f16(
                                            A_rmem[:, :, 0:4, kr:kr+1, 0:2],
                                            A_smem[(ks+1) % 2, mw * 64 + 0 : mw * 64 + 32, 8*kr:8*kr+8],
                                            nmat0=4, nmat1=1,
                                        )
                                        Sm80_ldmatrix_f16(
                                            A_rmem[:, :, 4:8, kr:kr+1, 0:2],
                                            A_smem[(ks+1) % 2, mw * 64 + 32 : mw * 64 + 64, 8*kr:8*kr+8],
                                            nmat0=4, nmat1=1,
                                        )
                                        Sm80_ldmatrix_f16(
                                            B_rmem[:, :, 0:4, kr:kr+1, 0:2],
                                            B_smem[(ks+1) % 2, nw * 64 + 0 : nw * 64 + 32, 8*kr:8*kr+8],
                                            nmat0=4, nmat1=1,
                                        )
                                        Sm80_ldmatrix_f16(
                                            B_rmem[:, :, 4:8, kr:kr+1, 0:2],
                                            B_smem[(ks+1) % 2, nw * 64 + 32 : nw * 64 + 64, 8*kr:8*kr+8],
                                            nmat0=4, nmat1=1,
                                        )

                                    for ms in seq(0, 4):
                                        for ns in seq(0, 8):
                                            Sm80_mma_m16n8(
                                                D_rmem[mw, nw, ms, ns, :, :],
                                                A_rmem[:, :, ms * 2 : ms * 2 + 2, :, :],
                                                B_rmem[:, :, ns, :, :],
                                                D=f32,
                                                A=f16,
                                                B=f16,
                                                K_pack=2,
                                            )

                        Fence(Sm80_generic, cuda_in_order)

                    for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                        for nw in cuda_threads(0, 2, unit=cuda_warp):
                            C_tile = C[
                                batch,
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


add_case(starter_double_smem_gemm)



RING = 3


@proc
def starter_ring_smem_gemm(L: size, M: size, N: size, K: size, C: f32[L, M, N] @ CudaGmemLinear, A: f16[L, M, K] @ CudaGmemLinear, B: f16[L, N, K] @ CudaGmemLinear):
    assert M % 128 == 0
    assert N % 128 == 0
    assert K % 64 == 0
    with CudaDeviceFunction(blockDim=128, blocks_per_sm=2):
        for batch in cuda_tasks(0, L):
            for m_task in cuda_tasks(0, (M + 127) / 128):
                for n_task in cuda_tasks(0, (N + 127) / 128):
                    D_rmem: f32[2, 2, 4, 8, 16, 8] @ Sm80_RmemMatrixD_m16n8
                    for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                        for nw in cuda_threads(0, 2, unit=cuda_warp):
                            for ms in seq(0, 4):
                                for ns in seq(0, 8):
                                    Sm80_mma_m16n8_zero(D_rmem[mw, nw, ms, ns, :, :], D=f32)

                    A_smem: f16[RING, 4, 128, 8] @ CudaSmemLinear
                    B_smem: f16[RING, 4, 128, 8] @ CudaSmemLinear
                    A_rmem: f16[2, 2, 8, 4, 2, 8, 2, 2] @ CudaRmemPacked32
                    B_rmem: f16[2, 2, 8, 4, 2, 8, 2, 2] @ CudaRmemPacked32
                    cg: barrier[128] @ CudaCommitGroup

                    for ks in seq(0, K / 32 + RING - 1):
                        A_tile = A[batch, m_task * 128 : m_task * 128 + 128, ks * 32 : ks * 32 + 32]
                        B_tile = B[batch, n_task * 128 : n_task * 128 + 128, ks * 32 : ks * 32 + 32]

                        for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                            for nw in cuda_threads(0, 2, unit=cuda_warp):
                                if ks >= RING - 1:
                                    for s in seq(0, 2, pragma_unroll=0):
                                        for kr in seq(0, 2, pragma_unroll=0):
                                            Sm80_ldmatrix_f16(
                                                A_rmem[mw, nw, :, :, 1, s*4:s*4+4, kr:kr+1, 0:2],
                                                A_smem[(ks - RING + 1) % RING, 2 + kr, mw * 64 + 32 * s : mw * 64 + 32 * s + 32, :],
                                                nmat0=4, nmat1=1,
                                            )
                                            Sm80_ldmatrix_f16(
                                                B_rmem[mw, nw, :, :, 1, s*4:s*4+4, kr:kr+1, 0:2],
                                                B_smem[(ks - RING + 1) % RING, 2 + kr, nw * 64 + 32 * s : nw * 64 + 32 * s + 32, :],
                                                nmat0=4, nmat1=1,
                                            )
                                    for ms in seq(0, 4, pragma_unroll=0):
                                        for ns in seq(0, 8, pragma_unroll=0):
                                            Sm80_mma_m16n8(
                                                D_rmem[mw, nw, ms, ns, :, :],
                                                A_rmem[mw, nw, :, :, 0, ms * 2 : ms * 2 + 2, :, :],
                                                B_rmem[mw, nw, :, :, 0, ns, :, :],
                                                D=f32,
                                                A=f16,
                                                B=f16,
                                                K_pack=2,
                                            )

                        for tid in cuda_threads(0, 128, unit=cuda_thread):
                            for ki in seq(0, 4, pragma_unroll=0):
                                if ks < K / 32:
                                    Sm80_cp_async_1d(
                                        A_smem[ks % RING, ki, tid, :], A_tile[tid, ki * 8 : ki * 8 + 8],
                                        size0=8, dst=f16, src=f16
                                    )
                                    Sm80_cp_async_1d(
                                        B_smem[ks % RING, ki, tid, :], B_tile[tid, ki * 8 : ki * 8 + 8],
                                        size0=8, dst=f16, src=f16
                                    )
                            Arrive(Sm80_cp_async) >> cg[tid]
                            Await(cg[tid], cuda_in_order, RING - 2)
                        Fence(cuda_in_order, cuda_in_order)

                        for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                            for nw in cuda_threads(0, 2, unit=cuda_warp):
                                if ks >= RING - 2:
                                    for s in seq(0, 2, pragma_unroll=0):
                                        for kr in seq(0, 2, pragma_unroll=0):
                                            Sm80_ldmatrix_f16(
                                                A_rmem[mw, nw, :, :, 0, s*4:s*4+4, kr:kr+1, 0:2],
                                                A_smem[(ks - RING + 2) % RING, kr, mw * 64 + 32 * s : mw * 64 + 32 * s + 32, :],
                                                nmat0=4, nmat1=1,
                                            )
                                            Sm80_ldmatrix_f16(
                                                B_rmem[mw, nw, :, :, 0, s*4:s*4+4, kr:kr+1, 0:2],
                                                B_smem[(ks - RING + 2) % RING, kr, nw * 64 + 32 * s : nw * 64 + 32 * s + 32, :],
                                                nmat0=4, nmat1=1,
                                            )
                                if ks >= RING - 1:
                                    for ms in seq(0, 4, pragma_unroll=0):
                                        for ns in seq(0, 8, pragma_unroll=0):
                                            Sm80_mma_m16n8(
                                                D_rmem[mw, nw, ms, ns, :, :],
                                                A_rmem[mw, nw, :, :, 1, ms * 2 : ms * 2 + 2, :, :],
                                                B_rmem[mw, nw, :, :, 1, ns, :, :],
                                                D=f32,
                                                A=f16,
                                                B=f16,
                                                K_pack=2,
                                            )

                    for tid in cuda_threads(0, 128):
                        Arrive(Sm80_cp_async) >> cg[tid]
                        Await(cg[tid], cuda_in_order, 0)
                    Fence(cuda_in_order, cuda_in_order)

                    for mw in cuda_threads(0, 2, unit=2 * cuda_warp):
                        for nw in cuda_threads(0, 2, unit=cuda_warp):
                            C_tile = C[
                                batch,
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


starter_ring_smem_gemm = simplify(starter_ring_smem_gemm)
add_case(starter_ring_smem_gemm)


import json
json.dump(cases, open(__file__ + ".json", "w"))
