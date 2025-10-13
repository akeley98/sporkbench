from __future__ import annotations
from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *


cases = []


def schedule_Sm90a_gemm():
    unsafe = False
    enable_split_k = False
    smem_M = 256
    smem_N = 64 * 3
    smem_K = 32
    ncta_M = 1  # TODO implement
    ncta_N = 1  # TODO implement

    # Derived constants
    wg_M = smem_M // 2
    wg_N = smem_N
    cluster_M = smem_M * ncta_M
    cluster_N = smem_N * ncta_N

    @proc
    def gemm(
            L: size, M: size, N: size, K_splits: size, K_cluster: size,
            A: f32[L, M, K_splits, K_cluster],  # Row-major (K-major)
            B: f32[L, N, K_splits, K_cluster],  # Column-major (K-major)
            C: f32[L, N, M],  # Column-major
    ):
        assert M % cluster_M == 0
        assert N % cluster_N == 0
        assert K_cluster % smem_K == 0
        if enable_split_k:
            for memset_batch in seq(0, L):
                for memset_n in seq(0, N):
                    for memset_m in seq(0, M):
                        C[memset_batch, memset_n, memset_m] = 0

        for batch in seq(0, L):
            for task_k in seq(0, K_splits):
                for m in seq(0, M):
                    for n in seq(0, N):
                        D_rmem: f32
                        D_rmem = 0
                        for k in seq(0, K_cluster):
                            D_rmem += A[batch, m, task_k, k] * B[batch, n, task_k, k]
                        if enable_split_k:
                            C[batch, n, m] += D_rmem
                        else:
                            C[batch, n, m] = D_rmem

    gemm = simplify(gemm)  # Get rid of enable_split_k if stmts.

    # Extract cursors to initial proc.
    loop_batch = gemm.find_loop("batch")
    loop_task_k = gemm.find_loop("task_k")
    loop_m = gemm.find_loop("m")
    loop_n = gemm.find_loop("n")
    loop_k = gemm.find_loop("k")
    D_rmem = gemm.find_alloc_or_arg("D_rmem")
    D_zero = gemm.find("D_rmem = 0")
    if enable_split_k:
        C_assign = gemm.find("_ += D_rmem")
    else:
        C_assign = gemm.find("_ = D_rmem")
    gap_before_main = loop_k.before()
    gap_after_main = loop_k.after()

    gemm = divide_loop(gemm, loop_m, cluster_M, ("task_m", "sub_task_m"), tail="guard")
    loop_task_m = gemm.forward(loop_m)
    loop_sub_task_m = loop_task_m.body()[0]
    gemm = divide_loop(gemm, loop_n, cluster_N, ("task_n", "sub_task_n"), tail="guard")
    loop_task_n = gemm.forward(loop_n)
    loop_sub_task_n = loop_task_n.body()[0]

    # Move task_n loop to be the outer most loop.
    gemm = lift_scope(gemm, loop_task_n)
    gemm = lift_scope(gemm, loop_task_n)
    gemm = lift_scope(gemm, loop_task_n)

    # Generate CTA loops.
    # These are supposed to be perfect, because the inner loop from the
    # task_m/task_n have constant bounds cluster_M, cluster_N.
    gemm = divide_loop(gemm, loop_sub_task_m, smem_M, ("cta_m", "sub_cta_m"), perfect=True)
    loop_cta_m = gemm.forward(loop_sub_task_m)
    loop_sub_cta_m = loop_cta_m.body()[0]
    gemm = divide_loop(gemm, loop_sub_task_n, smem_N, ("cta_n", "sub_cta_n"), perfect=True)
    loop_cta_n = gemm.forward(loop_sub_task_n)
    loop_sub_cta_n = loop_cta_n.body()[0]

    # Move cta_n loop outside for/if to be just under cta_m loop.
    gemm = lift_scope(gemm, loop_cta_n)
    gemm = lift_scope(gemm, loop_cta_n)

    # expand dim of D_rmem so each iteration uses its own D_rmem.
    # This enables future parallelization.
    gemm = expand_dim(gemm, D_rmem, smem_N, "sub_cta_n")
    gemm = expand_dim(gemm, D_rmem, smem_M, "sub_cta_m")
    gemm = expand_dim(gemm, D_rmem, ncta_N, "cta_n")
    gemm = expand_dim(gemm, D_rmem, ncta_M, "cta_m")
    D_rmem = gemm.forward(D_rmem)

    # Set up the main loop.
    # First we have to lift D_rmem out, then fission out the
    # zero prologue and GMEM-write epilogue.
    # Divide K loop to yield main loop (k_iter)
    # Move k_iter loop to be just under the tasks loops.
    #
    # TODO allow non-perfect.
    # This is harder than for M/N, since we have to think about how
    # zero padding makes the extra K loads safe (D += 0 is no-op).
    gemm = divide_loop(gemm, loop_k, smem_K, ("k_iter", "k_sub_iter"), perfect=True)
    loop_k_iter = gemm.forward(loop_k)
    loop_k_sub_iter = loop_k_iter.body()[0]
    k_lifts = 0
    parent = loop_k_iter.parent()
    while True:
        if isinstance(parent, ForCursor):
            if parent.name() in ("task_m", "task_n"):
                break
        k_lifts += 1
        parent = parent.parent()
    for i in range(k_lifts):
        gemm = lift_alloc(gemm, D_rmem)
    gemm = fission(gemm, gap_before_main, n_lifts=k_lifts, unsafe_disable_checks=unsafe)
    gemm = fission(gemm, gap_after_main, n_lifts=k_lifts, unsafe_disable_checks=unsafe)
    for i in range(k_lifts):
        gemm = lift_scope(gemm, loop_k_iter)
    gap_before_main = gemm.forward(gap_before_main)
    gap_after_main = gemm.forward(gap_after_main)
    D_rmem = gemm.forward(D_rmem)
    loop_k_iter = gemm.forward(loop_k_iter)

    # Finalize zero prologue. TODO set cuda_threads loop, CudaWarps.
    D_zero = gemm.forward(D_zero)
    zero_m_loop = D_zero.parent().parent().parent().parent()  # TODO better way?
    gemm = divide_loop(gemm, zero_m_loop, wg_M, ("wg_m", "sub_wg_m"), perfect=True)
    sub_wg_m_loop = gemm.forward(zero_m_loop).body()[0]
    if False:
        # TODO can't unify due to guards
        gemm = replace(gemm, sub_wg_m_loop, Sm90_zero_scale_d_f32(M=wg_M, N=wg_N))

    # Finalize write-to-C epilogue.
    # Can't do this yet due to if guards inside the instr.
    if enable_split_k:
        assert 0
    else:
        C_assign = gemm.forward(C_assign)
        assign_m_loop = C_assign.parent().parent().parent().parent()  # TODO better way?
        gemm = divide_loop(gemm, assign_m_loop, wg_M, ("wg_m", "sub_wg_m"), perfect=True)
        sub_wg_m_loop = gemm.forward(assign_m_loop).body()[0]
        gemm = replace(gemm, sub_wg_m_loop, Sm90_mma_store_d_col_major_tf32(M=wg_M, N=wg_N))

    # Substitute cuda memset for 0-init.
    if enable_split_k:
        gemm = replace(gemm, gemm.find_loop("memset_batch"), cudaMemsetAsync0_3f32())
    print(gemm)
    return gemm


foo = schedule_Sm90a_gemm()
del foo


@proc
def dummy_cuda():
    with CudaDeviceFunction(blockDim=32):
        for task in cuda_tasks(0, 1):
            pass


import json
json.dump(cases, open(__file__ + ".json", "w"))
