from __future__ import annotations

import time

from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.platforms.cuda_tk import *

from exo.scalars import e4m3, e5m2, bf16, f16, f32

from typing import List

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
                    D_rmem : D_type[ncta_M, ncta_N, 2, 4, wg_M/64, 16, wg_N] @ Sm90_TkRmemTileD(wg_N)

                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    for ms in seq(0, wg_M / 64, pragma_unroll=0):
                                        Sm90_tk_zero_scale_d(D_rmem[cta_m,cta_n,wg_m,:,ms,:,:], N=wg_N, D=f32)

                    raw : barrier[ncta_M, ncta_N] @ CudaMbarrier
                    war : barrier(raw)[ncta_M, ncta_N] @ CudaMbarrier
                    cg : barrier[ncta_M, ncta_N, 2] @ CudaCommitGroup

                    A_smem : A_type[ncta_M, ncta_N, RING, tile_M, smem_K] @ Sm90_SmemSwizzled(128)
                    B_smem : B_type[ncta_M, ncta_N, RING, tile_N, smem_K] @ Sm90_SmemSwizzled(128)

                    # This loop should be cut at 1.
                    for iter_k in seq(0, (cluster_K + smem_K - 1) / smem_K):
                        with CudaWarps(0, 1, name="producer"):
                            for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                                for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                    Await(war[cta_m,cta_n], cuda_temporal, ~(RING-1))
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
                                ) >> raw[cta_m,:]
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
                                ) >> raw[:,cta_n]
                                for cta_m in cuda_threads(0, ncta_M, unit=cuda_cta_in_cluster):
                                    Arrive(cuda_temporal) >> raw[cta_m,:] >> raw[:,cta_n]
                        with CudaWarps(name="consumer"):
                            for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                                for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                    Await(raw[cta_m,cta_n], cuda_generic_and_async_proxy, ~0)

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

                                    Arrive(cuda_in_order) >> war[cta_m,:] >> war[:,cta_n]
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

cases = []

gemm_f32_f16_f16_m1n1 = make_Sm90a_generic_gemm(1, 1, f32, f16, f16, cases)
gemm_f32_f16_f16_m1n2 = make_Sm90a_generic_gemm(1, 2, f32, f16, f16, cases)
gemm_f32_f16_f16_m2n1 = make_Sm90a_generic_gemm(2, 1, f32, f16, f16, cases)
gemm_f32_f16_f16_m2n2 = make_Sm90a_generic_gemm(2, 2, f32, f16, f16, cases)

# bf16 and fp8 test, so file name is a misnomer...
gemm_f32_bf16_bf16_m1n1 = make_Sm90a_generic_gemm(1, 1, f32, bf16, bf16, cases)
gemm_f32_e4m3_e4m3_m1n1 = make_Sm90a_generic_gemm(1, 1, f32, e4m3, e4m3, cases)
gemm_f32_e5m2_e5m2_m1n1 = make_Sm90a_generic_gemm(1, 1, f32, e5m2, e5m2, cases)
gemm_f32_e5m2_e5m2_m2n1 = make_Sm90a_generic_gemm(2, 1, f32, e5m2, e5m2, cases)

import json
json.dump(cases, open(__file__ + ".json", "w"))
