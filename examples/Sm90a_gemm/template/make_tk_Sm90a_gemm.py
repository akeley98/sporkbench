# fmt: off
"""

DO NOT EDIT if you are in the Exo repository

Edit if you are in sporkbench

For now I'm planning to copy/paste the sporkbench dir into Exo.

"""

from __future__ import annotations

import time

from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.platforms.cuda_tk import *

from exo.scalars import f16, bf16, f32

from typing import List

from .Sm90a_gemm_pre_config import Sm90aGemmConfig


Sm90_multicast_copy_tensor_to_smem_swizzled_2f32 = Sm90_tma_load_multicast_2d.partial(dst=f32, src=f32)


def make_Sm90a_gemm(config: Sm90aGemmConfig, ncta_M: int, ncta_N: int, cases: List[dict]):
    assert isinstance(config.smem_M, int), "Need to import Sm90a_gemm_pre_config first and set config variables"
    assert isinstance(config.smem_N, int), "Need to import Sm90a_gemm_pre_config first and set config variables"
    assert isinstance(config.tma_to_gmem, int), "Need to import Sm90a_gemm_pre_config first and set config variables"
    assert isinstance(config.enable_split_k, int), "Need to import Sm90a_gemm_pre_config first and set config variables"
    my_warp_config = [
        CudaWarpConfig("producer", 4, setmaxnreg_dec=40),
        CudaWarpConfig("consumer", 8, setmaxnreg_inc=232),
    ]

    tma_to_gmem = bool(config.tma_to_gmem)
    enable_split_k = bool(config.enable_split_k)
    smem_M = config.smem_M
    smem_N = config.smem_N
    smem_K = 32
    assert smem_M % 128 == 0
    wg_M = smem_M // 2
    assert smem_N % 8 == 0
    assert 8 <= smem_N <= 256
    wg_N = smem_N
    RING = config.RING
    cluster_M = smem_M * ncta_M
    cluster_N = smem_N * ncta_N
    ping_pong = bool(config.ping_pong)
    cooperative = not ping_pong
    P_DEPTH = 2 if ping_pong else 1

    assert not ping_pong, "not implemented"

    tile_M = smem_M // P_DEPTH
    tile_N = smem_N

    if enable_split_k:
        assert tma_to_gmem
        assert not ping_pong

    # (batch dim, MN smem, k_task, K smem)
    smem_box_A = (1, tile_M // ncta_N, 1, smem_K)
    smem_box_B = (1, tile_N // ncta_M, 1, smem_K)  # ncta_M is not a typo
    # (batch dim, M smem, N smem)
    smem_box_C = (1, tile_M, tile_N)

    # K dimension of tensor is K_split * cluster_K
    # i.e. each task (cluster) is responsible for cluster_M * cluster_N * cluster_K
    # We divide the K dim into [K_split, cluster_K] as a workaround for Exo
    # quasi-affine indexing restrictions.
    # Unfortunately, the caller has to manually pass cluster_K = K // K_split.
    @proc
    def p(
        L: size, M: size, N: size, K_split: size, cluster_K: size,
        A: f32[L,M,K_split,cluster_K] @ CudaGmemLinear,
        B: f32[L,N,K_split,cluster_K] @ CudaGmemLinear,
        C: f32[L,M,N] @ CudaGmemLinear
    ):
        assert stride(A, 3) == 1
        assert stride(B, 3) == 1
        assert stride(C, 2) == 1
        assert L > 0
        assert M > 0
        assert N > 0
        assert cluster_K > 0
        assert cluster_K % 4 == 0
        assert M % cluster_M == 0  # TODO
        assert N % cluster_N == 0  # TODO

        A_tensorMap = A[:,:,:,:] @ Sm90_tensorMap(128, *smem_box_A)
        B_tensorMap = B[:,:,:,:] @ Sm90_tensorMap(128, *smem_box_B)
        C_tensorMap = C[:,:,:] @ Sm90_tensorMap(0, *smem_box_C)  # Only for tma_to_gmem=True

        if enable_split_k:
            cudaMemsetAsync0_3f32(L, N, M, C[:,:,:])

        with CudaDeviceFunction(clusterDim=ncta_M * ncta_N, warp_config=my_warp_config, blocks_per_sm=1):
          for batch in cuda_tasks(0, L):
            for task_k in cuda_tasks(0, K_split):
              for task_n in cuda_tasks(0, (N + cluster_N - 1) / cluster_N):
                for task_m in cuda_tasks(0, (M + cluster_M - 1) / cluster_M):
                    # For ping-pong TMA case only.
                    # Must be declared early to avoid aliasing with A_smem, B_smem.
                    ping_C: f32[ncta_M, ncta_N, tile_N, tile_M] @ CudaSmemLinear

                    D_rmem: f32[ncta_M, ncta_N, 2, 4, wg_M/64, 16, wg_N] @ Sm90_TkRmemTileD(wg_N)

                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    for ms in seq(0, wg_M / 64, pragma_unroll=0):
                                        Sm90_tk_zero_scale_d(D_rmem[cta_m,cta_n,wg_m,:,ms,:,:], N=wg_N, D=f32)

                    war: barrier[ncta_M, ncta_N, P_DEPTH, (RING - 1 + (cluster_K + smem_K - 1) / smem_K) @ ring_buffer_by(RING),
                        ] @ CudaMbarrierPreArrive(RING - 1)
                    raw: barrier[ncta_M, ncta_N, P_DEPTH, (cluster_K + smem_K - 1) / smem_K @ ring_buffer_by(RING),
                        ].ring_guarded_by(war) @ CudaMbarrierPreArrive(0)
                    cg: barrier[ncta_M, ncta_N, 2] @ Sm90_WgmmaCommitGroup

                    A_smem: f32[ncta_M, ncta_N, P_DEPTH, RING, tile_M, smem_K] @ Sm90_SmemSwizzled(128)
                    B_smem: f32[ncta_M, ncta_N, P_DEPTH, RING, tile_N, smem_K] @ Sm90_SmemSwizzled(128)

                    # This loop should be cut at 1
                    for iter_k in seq(0, (cluster_K + smem_K - 1) / smem_K):
                        with CudaWarps(name="producer"):
                          # TMA producer warp(s)
                          for pm in cuda_threads(0, P_DEPTH, unit=cuda_warp):
                            for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                                for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                    Await(war[cta_m,cta_n,pm, iter_k], cuda_temporal, 0)
                                Sm90_multicast_copy_tensor_to_smem_swizzled_2f32(
                                    A_smem[cta_m,:,pm,iter_k % RING,:,:],
                                    A_tensorMap[
                                        batch,
                                        (ncta_M*task_m + cta_m) * smem_M + pm * tile_M:
                                        (ncta_M*task_m + cta_m) * smem_M + (pm + 1) * tile_M,
                                        task_k,
                                        iter_k * smem_K:
                                        iter_k * smem_K + smem_K],
                                    ncta=ncta_N, cta_stride=1, size0=tile_M, size1=smem_K, smem_box=smem_box_A
                                ) >> raw[cta_m,:,pm, iter_k]
                            for cta_n in cuda_threads(0, ncta_N, unit=ncta_M * cuda_cta_in_cluster_strided(ncta_N)):
                                Sm90_multicast_copy_tensor_to_smem_swizzled_2f32(
                                    B_smem[:,cta_n,pm,iter_k % RING,:,:],
                                    B_tensorMap[
                                        batch,
                                        (ncta_N*task_n+cta_n) * tile_N:
                                        (ncta_N*task_n+cta_n+1) * tile_N,
                                        task_k,
                                        iter_k * smem_K:
                                        iter_k * smem_K + smem_K],
                                    ncta=ncta_M, cta_stride=ncta_N, size0=tile_N, size1=smem_K, smem_box=smem_box_B
                                ) >> raw[:,cta_n,pm, iter_k]
                                for cta_m in cuda_threads(0, ncta_M, unit=cuda_cta_in_cluster):
                                    Arrive(cuda_temporal) >> raw[cta_m,:,pm, iter_k] >> raw[:,cta_n,pm, iter_k]

                        with CudaWarps(name="consumer"):
                            for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                                for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                    for pm in cuda_threads(0, P_DEPTH, unit=(2 // P_DEPTH) * cuda_warpgroup):
                                        Await(raw[cta_m,cta_n,pm, iter_k], cuda_generic_and_async_proxy, 0)

                                    for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                        Fence(wgmma_fence_1, wgmma_fence_2)
                                        for ms in seq(0, wg_M / 64, pragma_unroll=0):
                                            Sm90_tk_mma_row_col(
                                                D_rmem[cta_m, cta_n, wg_m, :, ms, :, :],
                                                A_smem[cta_m, cta_n, 0, iter_k % RING, (wg_m*wg_M): ((wg_m+1)*wg_M), :],
                                                B_smem[cta_m, cta_n, 0, iter_k % RING, :, :],
                                                D=f32, A=f32, B=f32, N=wg_N, K=32,
                                            )
                                        Arrive(wgmma_async) >> cg[cta_m,cta_n,wg_m]
                                        if iter_k >= 1:
                                            Await(cg[cta_m,cta_n,wg_m], cuda_in_order, 1)

                                    for pm in cuda_threads(0, P_DEPTH, unit=(2 // P_DEPTH) * cuda_warpgroup):
                                        Arrive(cuda_in_order
                                        ) >> war[cta_m,:,pm, iter_k+RING-1] >> war[:,cta_n,pm, iter_k+RING-1]

                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    Await(cg[cta_m,cta_n,wg_m], cuda_in_order, 0)

                    Fence(cuda_in_order, cuda_in_order)

                    C_smem: f32[ncta_M, ncta_N, tile_N / 32, tile_M, 32] @ Sm90_SmemSwizzled(128)
                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    for ms in seq(0, wg_M / 64):
                                        for w in cuda_threads(0, 4, unit=cuda_warp):
                                            cuda_tk_store_rs_inner_cols_32(
                                                C_smem[cta_m, cta_n, :, wg_m * wg_M + ms * 64 + w * 16: wg_m * wg_M + ms * 64 + w * 16 + 16, :],
                                                D_rmem[cta_m, cta_n, wg_m, w, ms, :, :],
                                                dst=f32,
                                                src=f32,
                                                rows=16,
                                                outer_cols=tile_N // 32,
                                            )
                            Fence(cuda_in_order, cuda_in_order)
                            with CudaWarps(name="consumer"):
                                C_tile = C[
                                    batch,
                                    cluster_M * task_m + tile_M * cta_m : cluster_M * task_m + tile_M * cta_m + tile_M,
                                    cluster_N * task_n + tile_N * cta_n : cluster_N * task_n + tile_N * cta_n + tile_N,
                                ]
                                for n in seq(0, tile_N / 32):
                                    for m in cuda_threads(0, 8, unit=cuda_warp):
                                        cuda_tk_store_sg(
                                            C_tile[m * (tile_M / 8) : m * (tile_M / 8) + tile_M / 8, n * 32 : n * 32 + 32],
                                            C_smem[cta_m, cta_n, n, m * (tile_M / 8) : m * (tile_M / 8) + tile_M / 8, :],
                                            size0=tile_M // 8,
                                            size1=32,
                                            dst=f32,
                                            src=f32,
                                        )

                    Fence(cuda_in_order, cuda_in_order)

                    # if tma_to_gmem and cooperative:
                    #     Fence(cuda_in_order, cuda_in_order)
                    #     C_smem: f32[ncta_M, ncta_N, tile_N, tile_M] @ CudaSmemLinear
                    #     for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                    #         for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                    #             with CudaWarps(name="consumer"):
                    #                 for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                    #                     Sm90_mma_store_d_col_major_tf32(
                    #                         wg_M, wg_N, C_smem[cta_m, cta_n, :,wg_m * wg_M: wg_m * wg_M + wg_M],
                    #                         D_rmem[cta_m,cta_n,wg_m,:,:], M=wg_M, N=wg_N)
                    #             Fence(cuda_in_order, cuda_generic_and_async_proxy)
                    #             with CudaWarps(0, 1, name="producer"):
                    #                 if enable_split_k:
                    #                     Sm90_reduce_tensor_to_gmem_linear_2f32(
                    #                         C_tensorMap[
                    #                             batch,
                    #                             (ncta_N*task_n + cta_n) * tile_N: (ncta_N*task_n + cta_n) * tile_N + tile_N,
                    #                             (ncta_M*task_m + cta_m) * tile_M: (ncta_M*task_m + cta_m) * tile_M + tile_M],
                    #                         C_smem[cta_m, cta_n, :, :],
                    #                         size0=tile_N, size1=tile_M, smem_box=smem_box_C,
                    #                     )
                    #                 else:
                    #                     Sm90_copy_tensor_to_gmem_linear_2f32(
                    #                         C_tensorMap[
                    #                             batch,
                    #                             (ncta_N*task_n + cta_n) * tile_N:
                    #                             (ncta_N*task_n + cta_n) * tile_N + tile_N,
                    #                             (ncta_M*task_m + cta_m) * tile_M:
                    #                             (ncta_M*task_m + cta_m) * tile_M + tile_M],
                    #                         C_smem[cta_m, cta_n, :, :],
                    #                         size0=tile_N, size1=tile_M, smem_box=smem_box_C,
                    #                     )
                    #                 tma_cg: barrier @ CudaCommitGroup
                    #                 Arrive(tma_to_gmem_async) >> tma_cg
                    #                 Await(tma_cg, cuda_in_order, 0)
                    #     Fence(cuda_in_order, cuda_in_order)  # cluster scope
                    # elif tma_to_gmem and ping_pong:
                    #     cross_task_protect: barrier[ncta_M, ncta_N, 2] @ CudaMbarrier
                    #     cross_task_protect_B: barrier(cross_task_protect)[ncta_M, ncta_N, 2] @ CudaMbarrier
                    #     for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                    #         for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                    #             with CudaWarps(name="consumer"):
                    #                 for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                    #                     Await(cross_task_protect_B[cta_m, cta_n, wg_m], cuda_temporal, ~8)
                    #                     Arrive(cuda_in_order) >> cross_task_protect[cta_m, :, wg_m] >> cross_task_protect[:, cta_n, wg_m]

                    #                 _0to1: barrier @ CudaMbarrier
                    #                 _1to0: barrier(_0to1) @ CudaMbarrier

                    #                 # Breaks sync-check:
                    #                 # These Await(s) need to be inside the wg_m loop.
                    #                 # Currently Await(_1to0) happens "before" Arrive >> _1to0
                    #                 # which should be a "no forward progress guarantee" error anyway.
                    #                 # Regardless, SMEM free check wouldn't have passed anyway
                    #                 # due to cluster-wide overapproximation.
                    #                 with CudaWarps(0, 4, name="consumer"):
                    #                     Await(_1to0, cuda_in_order, ~1)
                    #                 with CudaWarps(4, 8, name="consumer"):
                    #                     Await(_0to1, cuda_in_order, ~0)

                    #                 for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                    #                     Sm90_mma_store_d_col_major_tf32(
                    #                         wg_M, wg_N, ping_C[cta_m, cta_n, :, :], D_rmem[cta_m, cta_n, wg_m,:,:],
                    #                         M=wg_M, N=wg_N
                    #                     )
                    #                     # One warp waits for above SMEM write to finish,
                    #                     # then uses TMA to copy to GMEM. Signal cross-warpgroup
                    #                     # barrier afterwards so other warpgroup can use ping_C.
                    #                     intra_wg: barrier @ CudaMbarrier
                    #                     Arrive(cuda_in_order) >> intra_wg
                    #                     Await(intra_wg, cuda_generic_and_async_proxy, ~0)
                    #                     with CudaWarps(0, 1, name="consumer"):
                    #                         Sm90_copy_tensor_to_gmem_linear_2f32(
                    #                             C_tensorMap[
                    #                                 batch,
                    #                                 (ncta_N*task_n + cta_n) * tile_N:
                    #                                 (ncta_N*task_n + cta_n) * tile_N + tile_N,
                    #                                 (2*ncta_M*task_m + 2*cta_m + wg_m) * tile_M:
                    #                                 (2*ncta_M*task_m + 2*cta_m + wg_m) * tile_M + tile_M],
                    #                             ping_C[cta_m, cta_n, :, :],
                    #                             size0=tile_N, size1=tile_M, smem_box=smem_box_C,
                    #                         )

                    #                         tma_cg: barrier @ CudaCommitGroup
                    #                         Arrive(tma_to_gmem_async) >> tma_cg
                    #                         Await(tma_cg, cuda_in_order, 0)

                    #                 with CudaWarps(0, 4, name="consumer"):
                    #                     Arrive(cuda_in_order) >> _0to1
                    #                 with CudaWarps(4, 8, name="consumer"):
                    #                     Arrive(cuda_in_order) >> _1to0

                    #             # Total off-spec usage.
                    #             # Wait until consumer warpgroup is done reading SMEM before
                    #             # allowing corresponding producer warp to execute,
                    #             # when the current CUDA Cluster moves on to the next task
                    #             # (persistent kernel). This needs to consider cluster CTAs
                    #             # due to TMA multicast.
                    #             # cross_task_protect_B is just for barrier guarding requirements.
                    #             with CudaWarps(name="producer"):
                    #                 for w in cuda_threads(0, 2, unit=cuda_warp):
                    #                     Await(cross_task_protect[cta_m, cta_n, w], cuda_temporal, ~0)
                    #                     Arrive(cuda_temporal) >> cross_task_protect_B[cta_m, cta_n, w]
                    # else:
                    #     # Await(cg, cuda_in_order, 0) and write D_rmem -> C steps are fissioned.
                    #     # We must not arrive on the epilogue cluster sync until all wgmma retire.
                    #     cluster_sync: barrier @ CudaClusterSync
                    #     Arrive(cuda_in_order) >> cluster_sync

                    #     for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                    #         for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                    #             with CudaWarps(name="consumer"):
                    #                 for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                    #                     Sm90_mma_store_d_col_major_tf32(
                    #                         M - ((ncta_M*task_m + cta_m) * smem_M + wg_m * wg_M),
                    #                         N - (ncta_N*task_n + cta_n) * tile_N,
                    #                         C[batch,
                    #                           (ncta_N*task_n + cta_n) * tile_N
                    #                         : ((ncta_N*task_n + cta_n)+1) * tile_N,
                    #                           (ncta_M*task_m + cta_m) * smem_M + wg_m * wg_M
                    #                         : (ncta_M*task_m + cta_m) * smem_M + wg_m * wg_M + wg_M],
                    #                         D_rmem[cta_m,cta_n,wg_m,:,:], M=wg_M, N=wg_N)

                    #     Await(cluster_sync, cuda_in_order, 0)

    if tma_to_gmem and ping_pong:
        assert not enable_split_k
    # Remove ping_C or C_tensorMap on code paths that don't use them.
    else:
        p = delete_buffer(p, "ping_C")
    if not tma_to_gmem:
        p = inline_window(p, "C_tensorMap = _")

    # Give unique name and specialize 0th k-iter due to scale_d ptxas issues.
    p = rename(p, config.make_proc_name(ncta_M, ncta_N))
    p = cut_loop(p, p.find_loop("iter_k"), 1)
    p = simplify(p)

    # Timed sync check
    t = time.time()
    if False:
        print("NO SYNC CHECK: %s" % (p.name(),))
        # p._hack_no_smem_free_check = True
    else:
        K_split = 2 if enable_split_k else 1
        p.sync_check(L=2, M=500, N=800, cluster_K=240, K_split=K_split)
    dt = time.time() - t
    print("%.3f s, %s" % (dt, p.name()))

    # sporkbench cases
    if cases is not None:
        j_case = {
            "algorithm": "gemm",
            "A_type": "f32",
            "B_type": "f32",
            "C_type": "f32",
            "M_divisor": cluster_M,
            "N_divisor": cluster_N,
            "proc": p.name(),
            "args": ["L", "M", "N", "K_split", "K_cluster", "A", "B", "C"],
            "A_major": "row", "B_major": "col", "C_major": "row",
        }
        if not enable_split_k:
            j_case["K_split_max"] = 1
        cases.append(j_case)

    return p


def make_Sm90a_generic_gemm(ncta_M: int, ncta_N: int, D_type, A_type, B_type, cases: List[dict]):
    # K_split added for API compatibility; for now, K_split=1.

    assert D_type == f32, f"{D_type} needs to be f32 for now"
    smem_M = 128
    smem_N = 256
    smem_K = 128 * 8 // A_type.bits
    assert A_type.bits == B_type.bits, f"{A_type}, {B_type}"
    wg_M = smem_M // 2
    wg_N = smem_N
    cluster_M = smem_M * ncta_M
    cluster_N = smem_N * ncta_N
    tile_M = smem_M
    tile_N = smem_N
    RING = 4

    # (batch dim, MN smem, k_task, K smem)
    smem_box_A = (1, tile_M // ncta_N, 1, smem_K)
    smem_box_B = (1, tile_N // ncta_M, 1, smem_K)  # ncta_M is not a typo
    # (batch dim, M smem, N smem)
    smem_box_C = (1, tile_M, tile_N)  # For C_tensorMap if needed.

    enable_split_k = False  # TODO

    my_warp_config = [
        CudaWarpConfig("producer", 4, setmaxnreg_dec=40),
        CudaWarpConfig("consumer", 8, setmaxnreg_inc=232),
    ]

    # K dimension of tensor is K_split * cluster_K
    # i.e. each task (cluster) is responsible for cluster_M * cluster_N * cluster_K
    # We divide the K dim into [K_split, cluster_K] as a workaround for Exo
    # quasi-affine indexing restrictions.
    # Unfortunately, the caller has to manually pass cluster_K = K // K_split.
    @proc
    def p(
        L: size, M: size, N: size, K_split: size, cluster_K: size,
        A: A_type[L,M,K_split,cluster_K] @ CudaGmemLinear,
        B: B_type[L,N,K_split,cluster_K] @ CudaGmemLinear,
        C: D_type[L,M,N] @ CudaGmemLinear,
    ):
        assert stride(A, 3) == 1
        assert stride(B, 3) == 1
        assert stride(C, 2) == 1
        assert L > 0
        assert M > 0
        assert N > 0
        assert cluster_K > 0
        assert cluster_K % 4 == 0
        assert M % cluster_M == 0  # TODO
        assert N % cluster_N == 0  # TODO

        A_tensorMap = A[:,:,:,:] @ Sm90_tensorMap(128, *smem_box_A)
        B_tensorMap = B[:,:,:,:] @ Sm90_tensorMap(128, *smem_box_B)

        with CudaDeviceFunction(clusterDim=ncta_M * ncta_N, warp_config=my_warp_config, blocks_per_sm=1):
          for batch in cuda_tasks(0, L):
            for task_k in cuda_tasks(0, K_split):
              for task_n in cuda_tasks(0, (N + cluster_N - 1) / cluster_N):
                for task_m in cuda_tasks(0, (M + cluster_M - 1) / cluster_M):
                    D_rmem: D_type[ncta_M, ncta_N, 2, 4, wg_M/64, 16, wg_N] @ Sm90_TkRmemTileD(wg_N)

                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    for ms in seq(0, wg_M / 64, pragma_unroll=0):
                                        Sm90_tk_zero_scale_d(D_rmem[cta_m,cta_n,wg_m,:,ms,:,:], N=wg_N, D=f32)

                    war: barrier[ncta_M, ncta_N, (RING - 1 + (cluster_K + smem_K - 1) / smem_K) @ ring_buffer_by(RING),
                        ] @ CudaMbarrierPreArrive(RING - 1)
                    raw: barrier[ncta_M, ncta_N, (cluster_K + smem_K - 1) / smem_K @ ring_buffer_by(RING),
                        ].ring_guarded_by(war) @ CudaMbarrierPreArrive(0)
                    cg: barrier[ncta_M, ncta_N, 2] @ Sm90_WgmmaCommitGroup

                    A_smem: A_type[ncta_M, ncta_N, RING, tile_M, smem_K] @ Sm90_SmemSwizzled(128)
                    B_smem: B_type[ncta_M, ncta_N, RING, tile_N, smem_K] @ Sm90_SmemSwizzled(128)

                    # This loop should be cut at 1.
                    for iter_k in seq(0, (cluster_K + smem_K - 1) / smem_K):
                        with CudaWarps(0, 1, name="producer"):
                            for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                                for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                    Await(war[cta_m,cta_n,iter_k], cuda_temporal, 0)
                                Sm90_tma_load_multicast_2d(
                                    A_smem[cta_m,:,iter_k % RING,:,:],
                                    A_tensorMap[
                                        batch,
                                        (ncta_M*task_m + cta_m) * smem_M:
                                        (ncta_M*task_m + cta_m) * smem_M + tile_M,
                                        task_k,
                                        iter_k * smem_K:
                                        iter_k * smem_K + smem_K],
                                    ncta=ncta_N, cta_stride=1, size0=tile_M, size1=smem_K,
                                    smem_box=smem_box_A, dst=A_type, src=A_type,
                                ) >> raw[cta_m,:,iter_k]
                            for cta_n in cuda_threads(0, ncta_N, unit=ncta_M * cuda_cta_in_cluster_strided(ncta_N)):
                                Sm90_tma_load_multicast_2d(
                                    B_smem[:,cta_n,iter_k % RING,:,:],
                                    B_tensorMap[
                                        batch,
                                        (ncta_N*task_n+cta_n) * smem_N:
                                        (ncta_N*task_n+cta_n) * smem_N + smem_N,
                                        task_k,
                                        iter_k * smem_K:
                                        iter_k * smem_K + smem_K],
                                    ncta=ncta_M, cta_stride=ncta_N, size0=tile_N, size1=smem_K,
                                    smem_box=smem_box_B, dst=B_type, src=B_type,
                                ) >> raw[:,cta_n,iter_k]
                                for cta_m in cuda_threads(0, ncta_M, unit=cuda_cta_in_cluster):
                                    Arrive(cuda_temporal) >> raw[cta_m,:,iter_k] >> raw[:,cta_n,iter_k]
                        with CudaWarps(name="consumer"):
                            for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                                for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                    Await(raw[cta_m,cta_n,iter_k], cuda_generic_and_async_proxy, 0)

                                    for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                        Fence(wgmma_fence_1, wgmma_fence_2)
                                        for ms in seq(0, wg_M / 64, pragma_unroll=0):
                                            Sm90_tk_mma_row_col(
                                                D_rmem[cta_m, cta_n, wg_m, :, ms, :, :],
                                                A_smem[cta_m, cta_n, iter_k % RING, (wg_m*wg_M): ((wg_m+1)*wg_M), :],
                                                B_smem[cta_m, cta_n, iter_k % RING, :, :],
                                                D=D_type, A=A_type, B=B_type, N=wg_N, K=smem_K,
                                            )
                                        Arrive(wgmma_async) >> cg[cta_m,cta_n,wg_m]
                                        if iter_k >= 1:
                                            Await(cg[cta_m,cta_n,wg_m], cuda_in_order, 1)

                                    Arrive(cuda_in_order
                                    ) >> war[cta_m,:, iter_k+RING-1] >> war[:,cta_n, iter_k+RING-1]
                    # end for iter_k

                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    Await(cg[cta_m,cta_n,wg_m], cuda_in_order, 0)

                    Fence(cuda_in_order, cuda_in_order)

                    C_smem: D_type[ncta_M, ncta_N, tile_N / 32, tile_M, 32] @ Sm90_SmemSwizzled(128)
                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    for ms in seq(0, wg_M / 64):
                                        for w in cuda_threads(0, 4, unit=cuda_warp):
                                            cuda_tk_store_rs_inner_cols_32(
                                                C_smem[cta_m, cta_n, :, wg_m * wg_M + ms * 64 + w * 16: wg_m * wg_M + ms * 64 + w * 16 + 16, :],
                                                D_rmem[cta_m, cta_n, wg_m, w, ms, :, :],
                                                dst=f32,
                                                src=f32,
                                                rows=16,
                                                outer_cols=tile_N // 32,
                                            )
                            Fence(cuda_in_order, cuda_in_order)
                            with CudaWarps(name="consumer"):
                                C_tile = C[
                                    batch,
                                    cluster_M * task_m + tile_M * cta_m : cluster_M * task_m + tile_M * cta_m + tile_M,
                                    cluster_N * task_n + tile_N * cta_n : cluster_N * task_n + tile_N * cta_n + tile_N,
                                ]
                                for n in seq(0, tile_N / 32):
                                    for m in cuda_threads(0, 8, unit=cuda_warp):
                                        cuda_tk_store_sg(
                                            C_tile[m * (tile_M / 8) : m * (tile_M / 8) + tile_M / 8, n * 32 : n * 32 + 32],
                                            C_smem[cta_m, cta_n, n, m * (tile_M / 8) : m * (tile_M / 8) + tile_M / 8, :],
                                            size0=tile_M // 8,
                                            size1=32,
                                            dst=f32,
                                            src=f32,
                                        )

                    Fence(cuda_in_order, cuda_in_order)

    # Give unique name and specialize 0th k-iter due to scale_d ptxas issues.
    p = rename(p, f"xgemm_Sm90a_{D_type}_{A_type}_{B_type}_m{ncta_M}n{ncta_N}")
    p = cut_loop(p, p.find_loop("iter_k"), 1)
    p = simplify(p)

    # Timed sync check
    t = time.time()
    if True:
        K_split = 2 if enable_split_k else 1
        p.sync_check(L=2, M=500, N=800, cluster_K=240, K_split=K_split)
        dt = time.time() - t
        print("%.3f s, %s" % (dt, p.name()))
    else:
        print("No sync check: %s" % (p.name(), ))

    # sporkbench cases
    if cases is not None:
        j_case = {
            "algorithm": "gemm",
            "A_type": str(A_type),
            "B_type": str(B_type),
            "C_type": str(D_type),
            "M_divisor": cluster_M,
            "N_divisor": cluster_N,
            "proc": p.name(),
            "args": ["L", "M", "N", "K_split", "K_cluster", "A", "B", "C"],
            "A_major": "row", "B_major": "col", "C_major": "row",
        }
        if not enable_split_k:
            j_case["K_split_max"] = 1
        cases.append(j_case)

    return p


def make_Sm90a_generic_gemm_Brow(
        ncta_M: int, ncta_N: int,
        D_type, A_type, B_type, C_type,
        A_mode: str, cases: List[dict],
        smem_N=256,
):
    # K_split added for API compatibility; for now, K_split=1.

    assert A_mode in ("rmem", "row")
    A_is_rmem = (A_mode == "rmem")

    smem_M = 128
    smem_K = 128 * 8 // A_type.bits
    assert A_type.bits == B_type.bits, f"{A_type}, {B_type}"
    assert B_type.bits == 16, f"{B_type} cannot be transposed"
    wg_M = smem_M // 2
    wg_N = smem_N
    cluster_M = smem_M * ncta_M
    cluster_N = smem_N * ncta_N
    tile_M = smem_M
    tile_N = smem_N
    RING = 4

    # (batch dim, M smem, k_task, K smem)
    smem_box_A = (1, tile_M // ncta_N, 1, smem_K)
    # (batch dim, k_task, K smem, N inner = 64)
    smem_box_B = (1, 1, smem_K // ncta_M, 64)
    # (batch dim, M smem, N smem)
    smem_box_C = (1, tile_M, tile_N)  # For C_tensorMap if needed.

    # Information needed to help us stage each warp's [16, wg_N]-sized D tile
    # into the swizzled C_smem used for the epilogue.
    epilogue_advice = cuda_tk_store_rs_advice(16, wg_N, dst=C_type, src=D_type, swizzle=128)
    C_inner_cols = epilogue_advice.swizzle_elements
    local_cuda_tk_store_rs = epilogue_advice.instr

    enable_split_k = False  # TODO

    my_warp_config = [
        CudaWarpConfig("producer", 4, setmaxnreg_dec=40),
        CudaWarpConfig("consumer", 8, setmaxnreg_inc=232),
    ]

    # K dimension of tensor is K_split * cluster_K
    # i.e. each task (cluster) is responsible for cluster_M * cluster_N * cluster_K
    # We divide the K dim into [K_split, cluster_K] as a workaround for Exo
    # quasi-affine indexing restrictions.
    # Unfortunately, the caller has to manually pass cluster_K = K // K_split.
    @proc
    def p(
        L: size, M: size, N: size, K_split: size, cluster_K: size,
        A: A_type[L,M,K_split,cluster_K] @ CudaGmemLinear,
        B: B_type[L,K_split,cluster_K,N] @ CudaGmemLinear,
        C: C_type[L,M,N] @ CudaGmemLinear,
    ):
        assert stride(A, 3) == 1
        assert stride(B, 3) == 1
        assert stride(C, 2) == 1
        assert L > 0
        assert M > 0
        assert N > 0
        assert cluster_K > 0
        assert cluster_K % 4 == 0
        assert M % cluster_M == 0  # TODO
        assert N % cluster_N == 0  # TODO

        A_tensorMap = A[:,:,:,:] @ Sm90_tensorMap(128, *smem_box_A)
        B_tensorMap = B[:,:,:,:] @ Sm90_tensorMap(128, *smem_box_B)

        with CudaDeviceFunction(clusterDim=ncta_M * ncta_N, warp_config=my_warp_config, blocks_per_sm=1):
          for batch in cuda_tasks(0, L):
            for task_k in cuda_tasks(0, K_split):
              for task_n in cuda_tasks(0, (N + cluster_N - 1) / cluster_N):
                for task_m in cuda_tasks(0, (M + cluster_M - 1) / cluster_M):
                    D_rmem: D_type[ncta_M, ncta_N, 2, 4, wg_M/64, 16, wg_N] @ Sm90_TkRmemTileD(wg_N)

                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    for ms in seq(0, wg_M / 64, pragma_unroll=0):
                                        Sm90_tk_zero_scale_d(D_rmem[cta_m,cta_n,wg_m,:,ms,:,:], N=wg_N, D=D_type)

                    war: barrier[ncta_M, ncta_N, (RING - 1 + (cluster_K + smem_K - 1) / smem_K) @ ring_buffer_by(RING),
                        ] @ CudaMbarrierPreArrive(RING - 1)
                    raw: barrier[ncta_M, ncta_N, (cluster_K + smem_K - 1) / smem_K @ ring_buffer_by(RING),
                        ].ring_guarded_by(war) @ CudaMbarrierPreArrive(0)
                    cg: barrier[ncta_M, ncta_N, 2] @ Sm90_WgmmaCommitGroup

                    A_smem: A_type[ncta_M, ncta_N, RING, tile_M, smem_K] @ Sm90_SmemSwizzled(128)
                    B_smem: B_type[ncta_M, ncta_N, RING, tile_N / 64, smem_K, 64] @ Sm90_SmemSwizzled(128)

                    # Distributed dims: [CTA m, CTA n, 2 warpgroups, 4 warps]
                    # Each warp holds (wg_M/64)-many [16, smem_K]-sized tiles.
                    A_rmem: A_type[ncta_M, ncta_N, 2, 4, wg_M/64, 16, smem_K] @ Sm90_TkRmemTileA(smem_K)

                    # This loop should be cut at 1.
                    for iter_k in seq(0, (cluster_K + smem_K - 1) / smem_K):
                        with CudaWarps(0, 1, name="producer"):
                            for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                                for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                    Await(war[cta_m,cta_n,iter_k], cuda_temporal, 0)
                                Sm90_tma_load_multicast_2d(
                                    A_smem[cta_m,:,iter_k % RING,:,:],
                                    A_tensorMap[
                                        batch,
                                        (ncta_M*task_m + cta_m) * smem_M:
                                        (ncta_M*task_m + cta_m) * smem_M + tile_M,
                                        task_k,
                                        iter_k * smem_K:
                                        iter_k * smem_K + smem_K],
                                    ncta=ncta_N, cta_stride=1, size0=tile_M, size1=smem_K,
                                    smem_box=smem_box_A, dst=A_type, src=A_type,
                                ) >> raw[cta_m,:,iter_k]
                            for cta_n in cuda_threads(0, ncta_N, unit=ncta_M * cuda_cta_in_cluster_strided(ncta_N)):
                                for sn_tma in seq(0, smem_N / 64):  # TODO remove need to unroll this loop.
                                    Sm90_tma_load_multicast_2d(
                                        B_smem[:,cta_n,iter_k % RING, sn_tma ,:,:],
                                        B_tensorMap[
                                            batch,
                                            task_k,
                                            iter_k * smem_K:
                                            iter_k * smem_K + smem_K,
                                            (ncta_N*task_n+cta_n) * smem_N + sn_tma * 64:
                                            (ncta_N*task_n+cta_n) * smem_N + sn_tma * 64 + 64],
                                        ncta=ncta_M, cta_stride=ncta_N, size0=smem_K, size1=64,
                                        smem_box=smem_box_B, dst=B_type, src=B_type,
                                    ) >> raw[:,cta_n,iter_k]
                                for cta_m in cuda_threads(0, ncta_M, unit=cuda_cta_in_cluster):
                                    Arrive(cuda_temporal) >> raw[cta_m,:,iter_k] >> raw[:,cta_n,iter_k]
                        # End CudaWarps(0, 1, name="producer")
                        with CudaWarps(name="consumer"):
                            for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                                for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                    Await(raw[cta_m,cta_n,iter_k], cuda_generic_and_async_proxy, 0)

                                    if A_is_rmem:
                                        # A in RMEM case
                                        # Totally not efficient; just testing for now.
                                        for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                            for w in cuda_threads(0, 4, unit=cuda_warp):
                                                for ms in seq(0, wg_M / 64, pragma_unroll=0):
                                                    cuda_tk_load_rs_inner_cols_64(
                                                        A_rmem[cta_m, cta_n, wg_m, w, ms, :, :],
                                                        A_smem[cta_m, cta_n,
                                                               iter_k % RING :
                                                               iter_k % RING + 1,
                                                               wg_m * wg_M + ms * 64 + w * 16 :
                                                               wg_m * wg_M + ms * 64 + w * 16 + 16,
                                                               :],
                                                        dst=A_type, src=A_type, rows=16, outer_cols=1,
                                                    )
                                            Fence(wgmma_fence_1, wgmma_fence_2)
                                            for ms in seq(0, wg_M / 64, pragma_unroll=0):
                                                Sm90_tk_mma_rmem_row(
                                                    D_rmem[cta_m, cta_n, wg_m, :, ms, :, :],
                                                    A_rmem[cta_m, cta_n, wg_m, :, ms, :, :],
                                                    B_smem[cta_m, cta_n, iter_k % RING, :, :, :],
                                                    D=D_type, A=A_type, B=B_type, N64=wg_N // 64, K=smem_K,
                                                )
                                            Arrive(wgmma_async) >> cg[cta_m,cta_n,wg_m]
                                            Await(cg[cta_m,cta_n,wg_m], cuda_in_order, 0)
                                        # End A in RMEM case
                                    else:
                                        # A in SMEM case (normal)
                                        for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                            Fence(wgmma_fence_1, wgmma_fence_2)
                                            for ms in seq(0, wg_M / 64, pragma_unroll=0):
                                                Sm90_tk_mma_row_row(
                                                    D_rmem[cta_m, cta_n, wg_m, :, ms, :, :],
                                                    A_smem[cta_m, cta_n, iter_k % RING, (wg_m*wg_M): ((wg_m+1)*wg_M), :],
                                                    B_smem[cta_m, cta_n, iter_k % RING, :, :, :],
                                                    D=D_type, A=A_type, B=B_type, N64=wg_N // 64, K=smem_K,
                                                )
                                            Arrive(wgmma_async) >> cg[cta_m,cta_n,wg_m]
                                            if iter_k >= 1:
                                                Await(cg[cta_m,cta_n,wg_m], cuda_in_order, 1)
                                        # End A in SMEM case

                                    Arrive(cuda_in_order
                                    ) >> war[cta_m,:, iter_k+RING-1] >> war[:,cta_n, iter_k+RING-1]
                        # End CudaWarps(name="consumer")
                    # end for iter_k

                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    Await(cg[cta_m,cta_n,wg_m], cuda_in_order, 0)

                    Fence(cuda_in_order, cuda_in_order)

                    C_smem: C_type[ncta_M, ncta_N, tile_N / C_inner_cols, tile_M, C_inner_cols] @ Sm90_SmemSwizzled(128)
                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    for ms in seq(0, wg_M / 64):
                                        for w in cuda_threads(0, 4, unit=cuda_warp):
                                            local_cuda_tk_store_rs(
                                                C_smem[cta_m, cta_n, :, wg_m * wg_M + ms * 64 + w * 16: wg_m * wg_M + ms * 64 + w * 16 + 16, :],
                                                D_rmem[cta_m, cta_n, wg_m, w, ms, :, :]
                                            )
                            Fence(cuda_in_order, cuda_in_order)
                            with CudaWarps(name="consumer"):
                                C_tile = C[
                                    batch,
                                    cluster_M * task_m + tile_M * cta_m : cluster_M * task_m + tile_M * cta_m + tile_M,
                                    cluster_N * task_n + tile_N * cta_n : cluster_N * task_n + tile_N * cta_n + tile_N,
                                ]
                                for n in seq(0, tile_N / C_inner_cols):
                                    for m in cuda_threads(0, 8, unit=cuda_warp):
                                        cuda_tk_store_sg(
                                            C_tile[m * (tile_M / 8) : m * (tile_M / 8) + tile_M / 8, n * C_inner_cols : n * C_inner_cols + C_inner_cols],
                                            C_smem[cta_m, cta_n, n, m * (tile_M / 8) : m * (tile_M / 8) + tile_M / 8, :],
                                            size0=tile_M // 8,
                                            size1=C_inner_cols,
                                            dst=C_type,
                                            src=C_type,
                                        )

                    Fence(cuda_in_order, cuda_in_order)

    # Give unique name and specialize 0th k-iter due to scale_d ptxas issues.
    p = rename(p, f"xgemm_Sm90a_{C_type}_{A_type}_{B_type}_m{ncta_M}n{ncta_N}_{A_mode}_row")
    p = simplify(p)
    p = unroll_loop(p, p.find_loop("sn_tma"))  # TODO should not be needed.
    p = cut_loop(p, p.find_loop("iter_k"), 1)

    if not A_is_rmem:
        p = delete_buffer(p, "A_rmem")

    # Timed sync check
    t = time.time()
    if True:
        K_split = 2 if enable_split_k else 1
        p.sync_check(L=2, M=500, N=800, cluster_K=240, K_split=K_split)
        dt = time.time() - t
        print("%.3f s, %s" % (dt, p.name()))
    else:
        print("No sync check: %s" % (p.name(), ))

    A_major = "col" if A_mode == "col" else "row"

    # sporkbench cases
    if cases is not None:
        j_case = {
            "algorithm": "gemm",
            "A_type": str(A_type),
            "B_type": str(B_type),
            "C_type": str(C_type),
            "M_divisor": cluster_M,
            "N_divisor": cluster_N,
            "proc": p.name(),
            "args": ["L", "M", "N", "K_split", "K_cluster", "A", "B", "C"],
            "A_major": A_major, "B_major": "row", "C_major": "row",
        }
        if not enable_split_k:
            j_case["K_split_max"] = 1
        cases.append(j_case)

    return p
