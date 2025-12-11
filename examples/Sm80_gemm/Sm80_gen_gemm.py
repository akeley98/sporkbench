from __future__ import annotations

import time

from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *

from Sm80_gemm_pre_config import config, Sm80GemmConfig

assert isinstance(config.smem_M, int), "Need to import Sm80_gemm_pre_config first and set config variables"
assert isinstance(config.smem_N, int), "Need to import Sm80_gemm_pre_config first and set config variables"
assert isinstance(config.smem_K, int), "Need to import Sm80_gemm_pre_config first and set config variables"
assert isinstance(config.warp_M, int), "Need to import Sm80_gemm_pre_config first and set config variables"
assert isinstance(config.warp_N, int), "Need to import Sm80_gemm_pre_config first and set config variables"
assert isinstance(config.blocks_per_sm, int), "Need to import Sm80_gemm_pre_config first and set config variables"
assert isinstance(config.enable_split_k, int), "Need to import Sm80_gemm_pre_config first and set config variables"

M_divisor = 16
N_divisor = 8

def make_Sm80_gemm(config: Sm80GemmConfig, *, use_mbarrier: bool):
    smem_M = config.smem_M
    smem_N = config.smem_N
    smem_K = config.smem_K
    warp_M = config.warp_M
    warp_N = config.warp_N

    assert smem_M % warp_M == 0
    assert smem_N % warp_N == 0
    assert smem_K % 4 == 0

    M_warps = smem_M // warp_M
    N_warps = smem_N // warp_N
    blockDim = 32 * M_warps * N_warps
    enable_split_k = bool(config.enable_split_k)
    blocks_per_sm = config.blocks_per_sm
    if blocks_per_sm == 1:
        if use_mbarrier:
            sync_name = "mbarrier"
            RING = config.mbarrier_ring
            LAG = config.mbarrier_lag
        else:
            sync_name = "doubleBuf"
            RING = 2
            LAG = 1
    else:
        sync_name = "singleBuf"
        assert not use_mbarrier
        RING = 1
        LAG = 0

    # Number of 16-byte words in the K dimension of SMEM.
    smem_16B_K = smem_K // 4
    # Precompute pattern for assigning "load smem" work to threads.
    # In one sequential iteration, each team loads a
    # [1, smem_team_size, 16 / sizeof(T)] tile into SMEM.
    smem_team_size = blockDim // smem_16B_K
    assert smem_M % smem_team_size == 0
    assert smem_N % smem_team_size == 0
    smem_M_ITERS = smem_M // smem_team_size
    smem_N_ITERS = smem_N // smem_team_size

    if config.swizzle == 0:
        smem_type = CudaSmemLinear
    else:
        smem_type = Sm90_SmemSwizzled(config.swizzle)

    @proc
    def p(
        L: size, M: size, N: size, K_splits: size, K_cta: size,
        A: f32[L, M, K_splits, K_cta] @ CudaGmemLinear,
        B: f32[L, N, K_splits, K_cta] @ CudaGmemLinear,
        C: f32[L, N, M] @ CudaGmemLinear
    ):
        assert M % M_divisor == 0
        assert N % N_divisor == 0
        assert K_cta % smem_K == 0

        if enable_split_k:
            cudaMemsetAsync0_3f32(L, N, M, C[:,:,:])

        with CudaDeviceFunction(blockDim=blockDim, blocks_per_sm=blocks_per_sm):
          for batch in cuda_tasks(0, L):
            for k_task in cuda_tasks(0, K_splits):
              for m_task in cuda_tasks(0, (M + smem_M - 1) / smem_M):
                for n_task in cuda_tasks(0, (N + smem_N - 1) / smem_N):
                  # Per CTA code
                  raw: barrier @ CudaMbarrier
                  war: barrier(raw) @ CudaMbarrier

                  # Tiles (ring buffered)
                  A_smem: f32[RING, smem_16B_K, smem_M, 4] @ smem_type
                  B_smem: f32[RING, smem_16B_K, smem_N, 4] @ smem_type

                  # Zero-out accumulator (warp code)
                  D_rmem: f32[M_warps, N_warps, warp_M / 16, warp_N / 8, 16, 8] @ Sm80_RmemMatrixD(16, 8)
                  for mw in cuda_threads(0, M_warps, unit=N_warps * cuda_warp):
                    for nw in cuda_threads(0, N_warps, unit=cuda_warp):
                      for ms in seq(0, warp_M / 16):
                        for ns in seq(0, warp_N / 8):
                          Sm80_mma_zero_d_tf32(D_rmem[mw, nw, ms, ns, :, :])

                  # K tiles loop
                  # Don't accum tiles in the first LAG-many iterations.
                  # Don't load tiles in the laste LAG-many iterations.
                  # LAG iteration delay between load and use.
                  for k_iter in seq(0, K_cta / smem_K + LAG):
                    # Load SMEM except on final LAG-many iterations.
                    if k_iter < K_cta / smem_K:
                      if use_mbarrier:
                        # Wait for ring buffer to be consumed; don't wait for first RING-many iterations
                        Await(war, cuda_temporal, ~RING)
                      for kt in cuda_threads(0, smem_16B_K, unit=smem_team_size * cuda_thread):
                        for lane in cuda_threads(0, smem_team_size):
                          for ms in seq(0, smem_M_ITERS):
                            if m_task * smem_M + ms * smem_team_size + lane < M:
                              Sm80_cp_async_f32(
                                A_smem[k_iter % RING, kt, ms * smem_team_size + lane, :],
                                A[
                                    batch,
                                    m_task * smem_M + ms * smem_team_size + lane,
                                    k_task,
                                    k_iter * smem_K + kt * 4 :
                                    k_iter * smem_K + kt * 4 + 4,
                                ], size=4
                              )
                          for ns in seq(0, smem_N_ITERS):
                            if n_task * smem_N + ns * smem_team_size + lane < N:
                              Sm80_cp_async_f32(
                                B_smem[k_iter % RING, kt, ns * smem_team_size + lane, :],
                                B[
                                    batch,
                                    n_task * smem_N + ns * smem_team_size + lane,
                                    k_task,
                                    k_iter * smem_K + kt * 4 :
                                    k_iter * smem_K + kt * 4 + 4,
                                ], size=4
                              )
                      if use_mbarrier:
                        Arrive(Sm80_cp_async, 1) >> raw

                    # Single-buffer-only synchronization.
                    # Loaded values above immediately used in MMA below.
                    if LAG == 0:
                      Fence(Sm80_generic, cuda_in_order)

                    # MMA except on first LAG-many iterations.
                    if k_iter >= LAG:
                      if use_mbarrier:
                        # Wait for ring buffer to be filled
                        Await(raw, cuda_in_order, ~0)

                      for mw in cuda_threads(0, M_warps, unit=N_warps * cuda_warp):
                        for nw in cuda_threads(0, N_warps, unit=cuda_warp):
                          # Load all A/B matrix tiles ahead of time.
                          A_rmem: f32[warp_M / 16, smem_K / 4, 16, 4] @ Sm80_RmemMatrixA(16, 4)
                          B_rmem: f32[warp_N / 8, smem_K / 4, 4, 8] @ Sm80_RmemMatrixB(8, 4)
                          for ks in seq(0, smem_K / 4):
                            for ms in seq(0, warp_M / 16):
                              Sm80_mma_load_a_row_major_tf32(
                                A_rmem[ms, ks, :, :],
                                A_smem[
                                    (k_iter - LAG) % RING,
                                    ks,
                                    mw * warp_M + ms * 16:
                                    mw * warp_M + ms * 16 + 16,
                                    :
                                ], K=4
                              )
                            for ns in seq(0, warp_N / 8):
                              Sm80_mma_load_b_col_major_tf32(
                                B_rmem[ns, ks, :, :],
                                B_smem[
                                    (k_iter - LAG) % RING,
                                    ks,
                                    nw * warp_N + ns * 8:
                                    nw * warp_N + ns * 8 + 8,
                                    :
                                ], K=4
                              )
                          # Accumulate to accumulators owned by this warp
                          for ms in seq(0, warp_M / 16):
                            for ns in seq(0, warp_N / 8):
                              for ks in seq(0, smem_K / 4):
                                Sm80_mma_tf32(
                                    D_rmem[mw, nw, ms, ns, :, :],
                                    A_rmem[ms, ks, :, :],
                                    B_rmem[ns, ks, :, :],
                                    K=4,
                                )
                        # End nw-warps loop
                      # End mw-warps loop
                      if use_mbarrier:
                        Arrive(cuda_in_order, 1) >> war

                    # Synchronization between iterations, if not using mbarriers.
                    if use_mbarrier:
                      pass
                    elif LAG == 0:
                      Fence(cuda_in_order, cuda_in_order)
                    else:
                      Fence(Sm80_generic, cuda_in_order)
                  # End k_iter-seq loop

                  # Write out accumulator
                  for mw in cuda_threads(0, M_warps, unit=N_warps * cuda_warp):
                    for nw in cuda_threads(0, N_warps, unit=cuda_warp):
                      for ns in seq(0, warp_N / 8):
                        for ms in seq(0, warp_M / 16):
                          if n_task * smem_N + nw * warp_N + ns * 8 < N:
                            if m_task * smem_M + mw * warp_M + ms * 16 < M:
                              if enable_split_k:
                                Sm80_mma_atomic_reduce_d_col_major_tf32(
                                  C[
                                    batch,
                                    n_task * smem_N + nw * warp_N + ns * 8 :
                                    n_task * smem_N + nw * warp_N + ns * 8 + 8,
                                    m_task * smem_M + mw * warp_M + ms * 16 :
                                    m_task * smem_M + mw * warp_M + ms * 16 + 16,
                                  ],
                                  D_rmem[mw, nw, ms, ns, :, :]
                                )
                              else:
                                Sm80_mma_store_d_col_major_tf32(
                                  C[
                                    batch,
                                    n_task * smem_N + nw * warp_N + ns * 8 :
                                    n_task * smem_N + nw * warp_N + ns * 8 + 8,
                                    m_task * smem_M + mw * warp_M + ms * 16 :
                                    m_task * smem_M + mw * warp_M + ms * 16 + 16,
                                  ],
                                  D_rmem[mw, nw, ms, ns, :, :]
                                )
                  if use_mbarrier:
                    # Epilogue fence when using mbarriers, to make re-use of
                    # SMEM for the next task safe (persistent kernel).
                    Fence(cuda_in_order, cuda_in_order)

    p = rename(p, config.make_proc_name(sync_name))
    if not use_mbarrier:
      p = delete_buffer(p, p.find_alloc_or_arg("war"))
      p = delete_buffer(p, p.find_alloc_or_arg("raw"))
    p = simplify(p)
    K_splits = 2 if enable_split_k else 1
    t = time.time()
    p.sync_check(L=2, M=160, N=320, K_cta=smem_M * 5, K_splits=K_splits)
    dt = time.time() - t
    print("%.3f s, %s" % (dt, p.name()))
    return p


cases = []


def add_case_helper(p: exo.Procedure):
    cases.append({
        "algorithm": "gemm",
        "proc": p.name(),
        "args": ["L", "M", "N", "K_split", "K_cluster", "A", "B", "C"],
        "K_split_max": 0x7fffffff if config.enable_split_k else 1,
        "A_major": "row", "B_major": "col", "C_major": "col",
        "M_divisor": M_divisor, "N_divisor": N_divisor, "K_cluster_divisor": config.smem_K,
    })


gemm_fence = make_Sm80_gemm(config, use_mbarrier=False)
add_case_helper(gemm_fence)

if config.blocks_per_sm == 1:
    gemm_mbarrier = make_Sm80_gemm(config, use_mbarrier=True)
    add_case_helper(gemm_mbarrier)
