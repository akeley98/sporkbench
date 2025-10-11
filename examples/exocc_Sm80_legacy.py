from __future__ import annotations
import exo
from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *

cases = []


gemv_blockDim = 32
gemv_M0 = gemv_blockDim // 8
gemv_max_K = 2048


@proc
def gemv_warp_coop_8_smem(
        M: size,
        K: size,
        d_A: f32[M, K] @ CudaGmemLinear,
        d_x: f32[K] @ CudaGmemLinear,
        d_y: f32[M] @ CudaGmemLinear):
    assert M % gemv_M0 == 0
    assert K % 8 == 0
    assert K <= gemv_max_K
    with CudaDeviceFunction(blockDim=gemv_blockDim, blocks_per_sm=32):
        for m1 in cuda_tasks(0, M / gemv_M0):
            shared_x: f32[gemv_max_K] @ CudaSmemLinear
            for i in seq(0, (K + gemv_blockDim - 1) / gemv_blockDim):
                for tid in cuda_threads(0, gemv_blockDim):
                    if i * gemv_blockDim + tid < K:
                        shared_x[i * gemv_blockDim + tid] = d_x[i * gemv_blockDim + tid]
            Fence(cuda_in_order, cuda_in_order)
            for m0 in cuda_threads(0, gemv_M0, unit=8 * cuda_thread):
                # Teams of 8 threads cooperate to fill one element of d_y[m]
                # where m = m1 * gemv_M0 + m0.
                # Each thread "owns" one spot in the [2, 2, 2] accumulator.
                # Each thread sums up every 8th element in d_A[m, :] and h_x[:]
                partial_sum: f32[2, 2, 2] @ CudaRmem
                for k4 in cuda_threads(0, 2, unit=4 * cuda_thread):
                    for k2 in cuda_threads(0, 2, unit=2 * cuda_thread):
                        for k1 in cuda_threads(0, 2, unit=1 * cuda_thread):
                            partial_sum[k4, k2, k1] = 0
                            for k8 in seq(0, K / 8):
                                partial_sum[k4, k2, k1] += (
                                    d_A[m1 * gemv_M0 + m0, k8*8 + k4*4 + k2*2 + k1]
                                  * shared_x[k8*8 + k4*4 + k2*2 + k1]
                                )
                # XOR shuffle + sum to get totals.
                tmp: f32[2, 2, 2] @ CudaRmem
                for k2 in cuda_threads(0, 2, unit=2 * cuda_threads_strided(2, 4)):
                    for k1 in cuda_threads(0, 2, unit=2 * cuda_threads_strided(1, 4)):
                        cuda_shfl_xor_sync_1f32_sum(
                            tmp[:, k2, k1], partial_sum[:, k2, k1], laneMask=4)
                for k4 in cuda_threads(0, 2, unit=4 * cuda_thread):
                    for k1 in cuda_threads(0, 2, unit=2 * cuda_threads_strided(1, 2)):
                        cuda_shfl_xor_sync_1f32_sum(
                            partial_sum[k4, :, k1], tmp[k4, :, k1], laneMask=2)
                for k4 in cuda_threads(0, 2, unit=4 * cuda_thread):
                    for k2 in cuda_threads(0, 2, unit=2 * cuda_thread):
                        cuda_shfl_xor_sync_1f32_sum(
                            tmp[k4, k2, :], partial_sum[k4, k2, :], laneMask=1)
                        # Nominate one thread to write the output.
                        # This is written strangely; we can't access tmp[0, 0, 0]
                        # directly, due to distributed memory deduction rules.
                        for k1 in cuda_threads(0, 2, unit=cuda_thread):
                            tmp_scalar: f32 @ CudaRmem
                            tmp_scalar = tmp[k4, k2, k1]
                            if k4 == 0:
                                if k2 == 0:
                                    if k1 == 0:
                                        d_y[m1 * gemv_M0 + m0] = tmp_scalar
            Fence(cuda_in_order, cuda_in_order)


gemv_warp_coop_8_smem = simplify(gemv_warp_coop_8_smem)
gemv_warp_coop_8_smem.sync_check(M=256, K=128)


cases.append({
    "proc": "gemv_warp_coop_8_smem",
    "algorithm": "gemv",
    "args": ["M", "K", "A", "x", "y"],
    "K_max": gemv_max_K,
    "K_divisor": 1,
    "M_divisor": gemv_M0,
})


@proc
def gemv_warp_coop_8(
        M: size,
        K: size,
        d_A: f32[M, K] @ CudaGmemLinear,
        d_x: f32[K] @ CudaGmemLinear,
        d_y: f32[M] @ CudaGmemLinear):
    assert M % gemv_M0 == 0
    assert K % 8 == 0
    assert K <= gemv_max_K
    with CudaDeviceFunction(blockDim=gemv_blockDim, blocks_per_sm=32):
        for m1 in cuda_tasks(0, M / gemv_M0):
            for m0 in cuda_threads(0, gemv_M0, unit=8 * cuda_thread):
                # Teams of 8 threads cooperate to fill one element of d_y[m]
                # where m = m1 * gemv_M0 + m0.
                # Each thread "owns" one spot in the [2, 2, 2] accumulator.
                # Each thread sums up every 8th element in d_A[m, :] and h_x[:]
                partial_sum: f32[2, 2, 2] @ CudaRmem
                for k4 in cuda_threads(0, 2, unit=4 * cuda_thread):
                    for k2 in cuda_threads(0, 2, unit=2 * cuda_thread):
                        for k1 in cuda_threads(0, 2, unit=1 * cuda_thread):
                            partial_sum[k4, k2, k1] = 0
                            for k8 in seq(0, K / 8):
                                partial_sum[k4, k2, k1] += (
                                    d_A[m1 * gemv_M0 + m0, k8*8 + k4*4 + k2*2 + k1]
                                  * d_x[k8*8 + k4*4 + k2*2 + k1]
                                )
                # XOR shuffle + sum to get totals.
                tmp: f32[2, 2, 2] @ CudaRmem
                for k2 in cuda_threads(0, 2, unit=2 * cuda_threads_strided(2, 4)):
                    for k1 in cuda_threads(0, 2, unit=2 * cuda_threads_strided(1, 4)):
                        cuda_shfl_xor_sync_1f32_sum(
                            tmp[:, k2, k1], partial_sum[:, k2, k1], laneMask=4)
                for k4 in cuda_threads(0, 2, unit=4 * cuda_thread):
                    for k1 in cuda_threads(0, 2, unit=2 * cuda_threads_strided(1, 2)):
                        cuda_shfl_xor_sync_1f32_sum(
                            partial_sum[k4, :, k1], tmp[k4, :, k1], laneMask=2)
                for k4 in cuda_threads(0, 2, unit=4 * cuda_thread):
                    for k2 in cuda_threads(0, 2, unit=2 * cuda_thread):
                        cuda_shfl_xor_sync_1f32_sum(
                            tmp[k4, k2, :], partial_sum[k4, k2, :], laneMask=1)
                        # Nominate one thread to write the output.
                        # This is written strangely; we can't access tmp[0, 0, 0]
                        # directly, due to distributed memory deduction rules.
                        for k1 in cuda_threads(0, 2, unit=cuda_thread):
                            tmp_scalar: f32 @ CudaRmem
                            tmp_scalar = tmp[k4, k2, k1]
                            if k4 == 0:
                                if k2 == 0:
                                    if k1 == 0:
                                        d_y[m1 * gemv_M0 + m0] = tmp_scalar


gemv_warp_coop_8 = simplify(gemv_warp_coop_8)
gemv_warp_coop_8.sync_check(M=256, K=128)


cases.append({
    "proc": "gemv_warp_coop_8",
    "algorithm": "gemv",
    "args": ["M", "K", "A", "x", "y"],
    "K_divisor": 1,
    "M_divisor": gemv_M0,
})


Mw = 96
Nw = 64

M1 = 192
N1 = 256  # Does not change gracefully

K0 = 16
MMA_K = 4


@proc
def xgemm_Sm80_fence(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[K,N] @ CudaGmemLinear, C: f32[M,N] @ CudaGmemLinear):
    assert M % M1 == 0
    assert N % N1 == 0
    assert K % K0 == 0

    with CudaDeviceFunction(blockDim = 256, blocks_per_sm = 1):
        for m2 in cuda_tasks(0, M / M1):
            for n2 in cuda_tasks(0, N / N1):
                # Per CTA code

                # Tiles (double buffered)
                A_smem : f32[2, M1, K0] @ CudaSmemLinear
                B_smem : f32[2, K0, N1] @ CudaSmemLinear

                # Zero-out accumulator (warp code)
                D_rmem : f32[M1/Mw, N1/Nw, Mw/16, Nw/8, 16, 8] @ Sm80_RmemMatrixD(16, 8)
                for mw in cuda_threads(0, M1/Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1/Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw/16):
                            for n_seq in seq(0, Nw/8):
                                Sm80_mma_zero_d_tf32(D_rmem[mw,nw,m_seq, n_seq,:,:])

                # K tiles loop, double buffered
                # Don't accum tile in first iteration.
                # Don't load tile in last iteration.
                # 1 iteration delay between load and use.
                for k1 in seq(0, K / K0 + 1):
                    if k1 < K / K0:
                        # Load A tile
                        for m1 in seq(0, M1 / 64):
                            for m0 in cuda_threads(0, 64, unit=4 * cuda_thread):
                                for k0 in cuda_threads(0, 4, unit=cuda_thread):
                                    Sm80_cp_async_f32(A_smem[k1 % 2, m1 * 64 + m0, 4 * k0 : 4 * k0 + 4],
                                                      A[m2 * M1 + m1 * 64 + m0,
                                                      k1 * K0 + k0 * 4 : k1 * K0 + k0 * 4 + 4], size=4)

                        # Load B tile
                        for k0_seq in seq(0, 4):
                            for k0_par in cuda_threads(0, 4, unit=64 * cuda_thread):
                                for n0 in cuda_threads(0, 64, unit=cuda_thread):
                                    Sm80_cp_async_f32(B_smem[k1 % 2, k0_seq * 4 + k0_par, 4 * n0 : 4 * n0 + 4],
                                                      B[k1 * K0 + k0_seq * 4 + k0_par,
                                                      n2 * N1 + 4 * n0 : n2 * N1 + 4 * n0 + 4], size=4)
                # for-k1 (K tiles) loop continues
                    if k1 > 0:
                        for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                            for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                                # Load all B matrix tiles ahead of time
                                B_rmem : f32[K0/MMA_K, Nw/8, MMA_K, 8] @ Sm80_RmemMatrixB(8, MMA_K)
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_b_tf32(B_rmem[k_seq,n_seq,:,:],
                                                             B_smem[1 - k1 % 2,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K,
                                                             nw*Nw + n_seq*8 : nw*Nw + (n_seq+1)*8], K=MMA_K)

                                for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                    # Load A matrix tiles needed for m iteration
                                    A_rmem : f32[K0/MMA_K, 16, MMA_K] @ Sm80_RmemMatrixA(16, MMA_K)
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_a_tf32(A_rmem[k_seq,:,0:MMA_K],
                                                             A_smem[1 - k1 % 2,
                                                             mw*Mw + m_seq*16 : mw*Mw + (m_seq+1)*16,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K], K=MMA_K)
                                    # Accumulate to tile of warp tiles owned by warp.
                                    for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                        for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                            Sm80_mma_tf32(D_rmem[mw,nw,m_seq,n_seq,:,:],
                                                          A_rmem[k_seq,:,0:MMA_K],
                                                          B_rmem[k_seq,n_seq,:,:], K=MMA_K)

                    # Sm80_generic sync-tl = (cuda_in_order | Sm80_cp_async)
                    Fence(Sm80_generic, Sm80_generic)

                # for-k1 (K tiles) loop ends

                # Write out accumulator
                for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                            for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                Sm80_mma_store_d_tf32(
                                    C[m2 * M1 + mw * Mw + m_seq * 16 : m2 * M1 + mw * Mw + (m_seq+1) * 16,
                                    n2 * N1 + nw * Nw + n_seq * 8 : n2 * N1 + nw * Nw + (n_seq+1) * 8],
                                    D_rmem[mw,nw,m_seq,n_seq,:,:])


xgemm_Sm80_fence = simplify(xgemm_Sm80_fence)
xgemm_Sm80_fence.sync_check(M=M1 * 2, N=N1 * 2, K=K0 * 10)


cases.append({
    "algorithm": "gemm",
    "proc": "xgemm_Sm80_fence",
    "args": ["M", "N", "K", "A", "B", "C"],
    "M_divisor": M1, "N_divisor": N1, "K_divisor": K0,
    "A_major": "row", "B_major": "row", "C_major": "row",
})


RING = 3
LAG = 1

@proc
def xgemm_Sm80_mbarrier(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[K,N] @ CudaGmemLinear, C: f32[M,N] @ CudaGmemLinear):
    assert M % M1 == 0
    assert N % N1 == 0

    cudaMemsetAsync0_2f32(M, N, C[:,:])

    with CudaDeviceFunction(blockDim = 256, blocks_per_sm = 1):
        for m2 in cuda_tasks(0, M / M1):
            for n2 in cuda_tasks(0, N / N1):
                # Per CTA code
                raw: barrier @ CudaMbarrier
                war: barrier(raw) @ CudaMbarrier

                # Tiles (ring buffer)
                A_smem : f32[RING, M1, K0] @ CudaSmemLinear
                B_smem : f32[RING, K0, N1] @ CudaSmemLinear

                # Zero-out accumulator (warp code)
                D_rmem : f32[M1/Mw, N1/Nw, Mw/16, Nw/8, 16, 8] @ Sm80_RmemMatrixD(16, 8)
                for mw in cuda_threads(0, M1/Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1/Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw/16):
                            for n_seq in seq(0, Nw/8):
                                Sm80_mma_zero_d_tf32(D_rmem[mw,nw,m_seq, n_seq,:,:])

                # K tiles loop, ring buffered
                # Don't accum tile in first LAG-many iterations.
                # Don't load tile in last LAG-many iterations.
                # LAG iteration delay between load and use.
                for k1 in seq(0, K / K0 + LAG):
                    if k1 < K / K0:
                        # Wait for ring buffer to be consumed; don't wait for first RING-many iterations
                        Await(war, Sm80_cp_async, ~RING)

                        # Load A tile
                        for m1 in seq(0, M1 / 64):
                            for m0 in cuda_threads(0, 64, unit=4 * cuda_thread):
                                for k0 in cuda_threads(0, 4, unit=cuda_thread):
                                    Sm80_cp_async_f32(A_smem[k1 % RING, m1 * 64 + m0, 4 * k0 : 4 * k0 + 4],
                                                      A[m2 * M1 + m1 * 64 + m0,
                                                      k1 * K0 + k0 * 4 : k1 * K0 + k0 * 4 + 4],
                                                      size=4)

                        # Load B tile
                        for k0_seq in seq(0, 4):
                            for k0_par in cuda_threads(0, 4, unit=64 * cuda_thread):
                                for n0 in cuda_threads(0, 64, unit=cuda_thread):
                                    Sm80_cp_async_f32(B_smem[k1 % RING, k0_seq * 4 + k0_par, 4 * n0 : 4 * n0 + 4],
                                                      B[k1 * K0 + k0_seq * 4 + k0_par,
                                                      n2 * N1 + 4 * n0 : n2 * N1 + 4 * n0 + 4], size=4)
                        Arrive(Sm80_cp_async, 1) >> raw
                # for-k1 (K tiles) loop continues
                    if k1 >= LAG:
                        # Wait for ring buffer to be filled
                        Await(raw, cuda_in_order, ~0)

                        for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                            for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                                # Load all B matrix tiles ahead of time
                                B_rmem : f32[K0/MMA_K, Nw/8, MMA_K, 8] @ Sm80_RmemMatrixB(8, MMA_K)
                                for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_b_tf32(B_rmem[k_seq,n_seq,:,:],
                                                             B_smem[(k1 - LAG) % RING,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K,
                                                             nw*Nw + n_seq*8 : nw*Nw + (n_seq+1)*8], K=MMA_K)

                                for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                                    # Load A matrix tiles needed for m iteration
                                    A_rmem : f32[K0/MMA_K, 16, MMA_K] @ Sm80_RmemMatrixA(16, MMA_K)
                                    for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                        Sm80_mma_load_a_tf32(A_rmem[k_seq,:,:],
                                                             A_smem[(k1 - LAG) % RING,
                                                             mw*Mw + m_seq*16 : mw*Mw + (m_seq+1)*16,
                                                             k_seq*MMA_K:(k_seq+1)*MMA_K], K=MMA_K)
                                    # Accumulate to tile of warp tiles owned by warp.
                                    for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                        for k_seq in seq(0, K0 / MMA_K, pragma_unroll=0):
                                            Sm80_mma_tf32(D_rmem[mw,nw,m_seq,n_seq,:,:],
                                                          A_rmem[k_seq,:,:],
                                                          B_rmem[k_seq,n_seq,:,:], K=MMA_K)
                        # Signal that it's safe to overwrite ring buffer entry
                        Arrive(cuda_in_order, 1) >> war
                # for-k1 (K tiles) loop ends

                # Write out accumulator
                for mw in cuda_threads(0, M1 / Mw, unit=(N1/Nw) * cuda_warp):
                    for nw in cuda_threads(0, N1 / Nw, unit=cuda_warp):
                        for m_seq in seq(0, Mw / 16, pragma_unroll=0):
                            for n_seq in seq(0, Nw / 8, pragma_unroll=0):
                                Sm80_mma_atomic_reduce_d_tf32(
                                    C[m2 * M1 + mw * Mw + m_seq * 16 : m2 * M1 + mw * Mw + (m_seq+1) * 16,
                                    n2 * N1 + nw * Nw + n_seq * 8 : n2 * N1 + nw * Nw + (n_seq+1) * 8],
                                    D_rmem[mw,nw,m_seq,n_seq,:,:])
                Fence(cuda_in_order, cuda_in_order)


xgemm_Sm80_mbarrier = simplify(xgemm_Sm80_mbarrier)
xgemm_Sm80_mbarrier.sync_check(M=M1, N=N1 * 2, K=1024)


cases.append({
    "algorithm": "gemm",
    "proc": "xgemm_Sm80_mbarrier",
    "args": ["M", "N", "K", "A", "B", "C"],
    "M_divisor": M1, "N_divisor": N1, "K_divisor": K0,
    "A_major": "row", "B_major": "row", "C_major": "row",
})


import json
json.dump(cases, open(__file__ + ".json", "w"))
