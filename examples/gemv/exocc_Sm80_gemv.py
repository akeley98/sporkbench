from __future__ import annotations
import exo
from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *


gemv_blockDim = 32
gemv_M0 = gemv_blockDim // 8
gemv_max_K = 2048


cases = []


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


import json
json.dump(cases, open(__file__ + ".json", "w"))
