from __future__ import annotations

import time

from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.platforms.cuda_tk import *

from exo.scalars import e4m3, e5m2, e8m0, bf16, f16, f32

from typing import List

def make_Sm90a_generic_gemm(ncta_M: int, ncta_N: int, D_type, A_type, B_type, A_mode: str, cases: List[dict]):
    # K_split added for API compatibility; for now, K_split=1.

    assert A_mode in ("rmem", "row")
    A_is_rmem = (A_mode == "rmem")

    smem_M = 128
    smem_N = 256
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
    epilogue_advice = cuda_tk_store_rs_advice(16, wg_N, dst=D_type, src=D_type, swizzle=128)
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
                                        Sm90_tk_zero_scale_d(D_rmem[cta_m,cta_n,wg_m,:,ms,:,:], N=wg_N, D=D_type)

                    raw : barrier[ncta_M, ncta_N] @ CudaMbarrier
                    war : barrier(raw)[ncta_M, ncta_N] @ CudaMbarrier
                    cg : barrier[ncta_M, ncta_N, 2] @ CudaCommitGroup

                    A_smem : A_type[ncta_M, ncta_N, RING, tile_M, smem_K] @ Sm90_SmemSwizzled(128)
                    B_smem : B_type[ncta_M, ncta_N, RING, tile_N / 64, smem_K, 64] @ Sm90_SmemSwizzled(128)

                    # Distributed dims: [CTA m, CTA n, 2 warpgroups, 4 warps]
                    # Each warp holds (wg_M/64)-many [16, smem_K]-sized tiles.
                    A_rmem: A_type[ncta_M, ncta_N, 2, 4, wg_M/64, 16, smem_K] @ Sm90_TkRmemTileA(smem_K)

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
                                    ) >> raw[:,cta_n]
                                for cta_m in cuda_threads(0, ncta_M, unit=cuda_cta_in_cluster):
                                    Arrive(cuda_temporal) >> raw[cta_m,:] >> raw[:,cta_n]
                        # End CudaWarps(0, 1, name="producer")
                        with CudaWarps(name="consumer"):
                            for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                                for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                                    Await(raw[cta_m,cta_n], cuda_generic_and_async_proxy, ~0)

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

                                    Arrive(cuda_in_order) >> war[cta_m,:] >> war[:,cta_n]
                        # End CudaWarps(name="consumer")
                    # end for iter_k

                    for cta_m in cuda_threads(0, ncta_M, unit=ncta_N * cuda_cta_in_cluster):
                        for cta_n in cuda_threads(0, ncta_N, unit=cuda_cta_in_cluster):
                            with CudaWarps(name="consumer"):
                                for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                                    Await(cg[cta_m,cta_n,wg_m], cuda_in_order, 0)

                    Fence(cuda_in_order, cuda_in_order)

                    C_smem: D_type[ncta_M, ncta_N, tile_N / C_inner_cols, tile_M, C_inner_cols] @ Sm90_SmemSwizzled(128)
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
                                            dst=D_type,
                                            src=D_type,
                                        )

                    Fence(cuda_in_order, cuda_in_order)

    # Give unique name and specialize 0th k-iter due to scale_d ptxas issues.
    p = rename(p, f"xgemm_Sm90a_{D_type}_{A_type}_{B_type}_m{ncta_M}n{ncta_N}_{A_mode}_row")
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

    A_major = "col" if A_mode == "col" else "row"

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
            "A_major": A_major, "B_major": "row", "C_major": "row",
        }
        if not enable_split_k:
            j_case["K_split_max"] = 1
        cases.append(j_case)

    return p

cases = []

gemm_m1n1_f32_bf16_rmem_row = make_Sm90a_generic_gemm(1, 1, f32, bf16, bf16, "rmem", cases)
gemm_m2n1_f32_bf16_rmem_row = make_Sm90a_generic_gemm(2, 1, f32, bf16, bf16, "rmem", cases)

gemm_m1n1_f32_f16 = make_Sm90a_generic_gemm(1, 1, f32, f16, f16, "row", cases)
# gemm_m1n2_f32_f16 = make_Sm90a_generic_gemm(1, 2, f32, f16, f16, "row", cases)
# gemm_m2n1_f32_f16 = make_Sm90a_generic_gemm(2, 1, f32, f16, f16, "row", cases)
# gemm_m2n2_f32_f16 = make_Sm90a_generic_gemm(2, 2, f32, f16, f16, "row", cases)

gemm_m1n1_f32_bf16 = make_Sm90a_generic_gemm(1, 1, f32, bf16, bf16, "row", cases)
# gemm_m1n2_f32_bf16 = make_Sm90a_generic_gemm(1, 2, f32, bf16, bf16, "row", cases)
gemm_m2n1_f32_bf16 = make_Sm90a_generic_gemm(2, 1, f32, bf16, bf16, "row", cases)
# gemm_m2n2_f32_bf16 = make_Sm90a_generic_gemm(2, 2, f32, bf16, bf16, "row", cases)

gemm_m1n1_f16_f16 = make_Sm90a_generic_gemm(1, 1, f16, f16, f16, "row", cases)
gemm_m1n2_f16_f16 = make_Sm90a_generic_gemm(1, 2, f16, f16, f16, "row", cases)
gemm_m2n1_f16_f16 = make_Sm90a_generic_gemm(2, 1, f16, f16, f16, "row", cases)
# gemm_m2n2_f16_f16 = make_Sm90a_generic_gemm(2, 2, f16, f16, f16, "row", cases)


import json
json.dump(cases, open(__file__ + ".json", "w"))
