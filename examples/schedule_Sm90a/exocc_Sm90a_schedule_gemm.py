from __future__ import annotations
from exo import *
from exo.stdlib.scheduling import *
from exo.stdlib.stdlib import auto_stage_mem
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *


cases = []


@instr("// placeholder arrive")
def placeholder_arrive():
    pass


@instr("// placeholder await")
def placeholder_await():
    pass


def schedule_Sm90a_gemm():
    unsafe = False
    enable_split_k = False
    smem_M = 256
    smem_N = 64 * 3
    smem_K = 32
    ncta_M = 2
    ncta_N = 2

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
        # assert K_cluster % K_smem == 0
        assert K_cluster % 4 == 0
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
    batch_loop = gemm.find_loop("batch")
    task_k_loop = gemm.find_loop("task_k")
    m_loop = gemm.find_loop("m")
    n_loop = gemm.find_loop("n")
    k_loop = gemm.find_loop("k")
    D_rmem = gemm.find_alloc_or_arg("D_rmem")
    D_zero = gemm.find("D_rmem = 0")
    if enable_split_k:
        C_assign = gemm.find("_ += D_rmem")
    else:
        C_assign = gemm.find("_ = D_rmem")
    gap_before_main = k_loop.before()
    gap_after_main = k_loop.after()

    # Set up cuda_tasks loops.
    gemm = set_loop_mode(gemm, batch_loop, CudaTasks)
    gemm = set_loop_mode(gemm, task_k_loop, CudaTasks)
    gemm = divide_loop(gemm, m_loop, cluster_M, ("task_m", "sub_task_m"), tail="guard")
    task_m_loop = gemm.forward(m_loop)
    sub_task_m_loop = task_m_loop.body()[0]
    gemm = set_loop_mode(gemm, task_m_loop, CudaTasks)
    gemm = divide_loop(gemm, n_loop, cluster_N, ("task_n", "sub_task_n"), tail="guard")
    task_n_loop = gemm.forward(n_loop)
    sub_task_n_loop = task_n_loop.body()[0]
    gemm = set_loop_mode(gemm, task_n_loop, CudaTasks)

    # Move task_n loop to be the outer most loop.
    gemm = lift_scope(gemm, task_n_loop)
    gemm = lift_scope(gemm, task_n_loop)
    gemm = lift_scope(gemm, task_n_loop)

    # Generate CTA loops.
    # These are supposed to be perfect, because the inner loop from the
    # task_m/task_n have constant bounds cluster_M, cluster_N.
    gemm = divide_loop(gemm, sub_task_m_loop, smem_M, ("cta_m", "sub_cta_m"), perfect=True)
    cta_m_loop = gemm.forward(sub_task_m_loop)
    gemm = set_loop_mode(gemm, cta_m_loop, CudaThreads(unit=ncta_N * cuda_cta_in_cluster))
    gemm = divide_loop(gemm, sub_task_n_loop, smem_N, ("cta_n", "sub_cta_n"), perfect=True)
    n_cta_loop = gemm.forward(sub_task_n_loop)
    gemm = set_loop_mode(gemm, n_cta_loop, CudaThreads(unit=cuda_cta_in_cluster))

    # Move cta_n loop outside for/if to be just under cta_m loop.
    gemm = lift_scope(gemm, n_cta_loop)
    gemm = lift_scope(gemm, n_cta_loop)

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
    # Divide K loop to yield main loop (iter_k)
    # Move iter_k loop to be just under the tasks loops.
    #
    # Non-perfect K loop:
    # This is harder than for M/N, since we have to think about how
    # zero padding makes the extra K loads safe (D += 0 is no-op).
    gemm = divide_loop(gemm, k_loop, smem_K, ("iter_k", "sub_iter_k"), tail="guard")
    iter_k_loop = gemm.forward(k_loop)
    sub_iter_k_loop = iter_k_loop.body()[0]
    k_lifts = 0
    parent = iter_k_loop.parent()
    while True:
        if isinstance(parent, ForCursor):
            if isinstance(parent.loop_mode(), CudaTasks):
                break
        k_lifts += 1
        parent = parent.parent()
    gemm = lift_alloc(gemm, D_rmem, n_lifts=k_lifts)
    gemm = fission(gemm, gap_before_main, n_lifts=k_lifts, unsafe_disable_checks=unsafe)
    gemm = fission(gemm, gap_after_main, n_lifts=k_lifts, unsafe_disable_checks=unsafe)
    for i in range(k_lifts):
        gemm = lift_scope(gemm, iter_k_loop)
    gap_before_main = gemm.forward(gap_before_main)
    gap_after_main = gemm.forward(gap_after_main)
    D_rmem = gemm.forward(D_rmem)
    iter_k_loop = gemm.forward(iter_k_loop)

    # Stage A_smem, B_smem tiles above sub_cta_m loop.
    # TODO how to get a more stable reference to the input cursor.
    # We can't rely on the old sub_cta_m cursor since it's forwarded to
    # the wrong loop after fission.
    # TODO sucky that we have to use f-strings here; PAST can't get local variables???
    sub_cta_m_main_loop = iter_k_loop.body()[0].body()[0].body()[0]
    gemm = stage_mem(gemm, sub_cta_m_main_loop,
        f"A[batch, "
        f"(task_m * {ncta_M} + cta_m) * {smem_M} : (task_m * {ncta_M} + cta_m + 1) * {smem_M}, "
        f"task_k, iter_k * {smem_K}: (iter_k + 1) * {smem_K}]",
        "A_smem")
    gemm = stage_mem(gemm, sub_cta_m_main_loop,
        f"B[batch, "
        f"(task_n * {ncta_N} + cta_n) * {smem_N} : (task_n * {ncta_N} + cta_n + 1) * {smem_N}, "
        f"task_k, iter_k * {smem_K}: (iter_k + 1) * {smem_K}]",
        "B_smem")
    A_smem = gemm.find_alloc_or_arg("A_smem")
    B_smem = gemm.find_alloc_or_arg("B_smem")

    # Lift SMEM tiles to cluster scope.
    # Generate one shard per CTA in cluster.
    # Divide M/N dimension by 8.
    gemm = simplify(gemm)  # stage_mem generates exprs too hard for Exo to understand...
    for smem_cursor in (A_smem, B_smem):
        gemm = divide_dim(gemm, smem_cursor, 0, 8)
        gemm = expand_dim(gemm, smem_cursor, ncta_N, "cta_n")
        gemm = expand_dim(gemm, smem_cursor, ncta_M, "cta_m")
        gemm = lift_alloc(gemm, smem_cursor, n_lifts=3)

    # Fission CTA loops before and after B_smem load.
    # TODO how to get these cursors more elegantly?
    sub_cta_m_main_loop = gemm.forward(sub_cta_m_main_loop)
    B_smem_loop = sub_cta_m_main_loop.prev()
    gemm = fission(gemm, B_smem_loop.before(), n_lifts=2, unsafe_disable_checks=unsafe)
    gemm = fission(gemm, B_smem_loop.after(), n_lifts=2, unsafe_disable_checks=unsafe)

    # We have to reorder the loops for the B_smem load to be cta_n, cta_m.
    # This is needed for multicasting.
    # CTAs with the same cta_m value will multicast the same tile of B @ GMEM.
    # Therefore, to substitute the instruction later, we need to have cta_m inner.
    # Unlike CPU Exo, transposing these parallel loops requires us to rewrite
    # the unit, to still have the same assignment of loop iters to CTAs (manual for now).
    B_smem_loop = gemm.forward(B_smem_loop)
    B_smem_cta_n_loop = B_smem_loop.parent()
    B_smem_cta_m_loop = B_smem_cta_n_loop.parent()
    print(B_smem_cta_m_loop)
    gemm = reorder_loops(gemm, B_smem_cta_m_loop)
    gemm = update_loop_mode(gemm, B_smem_cta_n_loop, unit=ncta_M * cuda_cta_in_cluster_strided(ncta_N))
    gemm = update_loop_mode(gemm, B_smem_cta_m_loop, unit=cuda_cta_in_cluster)

    # M-warpgroup loop (children should be replaced with wgmma later)
    gemm = divide_loop(gemm, sub_cta_m_main_loop, wg_M, ("wg_m", "sub_wg_m"), perfect=True)
    wg_m_main_loop = gemm.forward(sub_cta_m_main_loop)
    gemm = set_loop_mode(gemm, wg_m_main_loop, CudaThreads(unit=cuda_warpgroup))

    # Finalize zero prologue. TODO set cuda_threads loop, CudaWarps.
    D_zero = gemm.forward(D_zero)
    zero_m_loop = D_zero.parent().parent().parent().parent()  # TODO better way?
    gemm = divide_loop(gemm, zero_m_loop, wg_M, ("wg_m", "sub_wg_m"), perfect=True)
    gemm = set_loop_mode(gemm, zero_m_loop, CudaThreads(unit=cuda_warpgroup))
    sub_wg_m_loop = gemm.forward(zero_m_loop).body()[0]
    if False:
        # TODO can't unify due to guards
        gemm = replace(gemm, sub_wg_m_loop, Sm90_zero_scale_d_f32(M=wg_M, N=wg_N))

    # Finalize write-to-C epilogue.
    if enable_split_k:
        assert 0
    else:
        C_assign = gemm.forward(C_assign)
        assign_m_loop = C_assign.parent().parent().parent().parent()  # TODO better way?
        gemm = divide_loop(gemm, assign_m_loop, wg_M, ("wg_m", "sub_wg_m"), perfect=True)
        gemm = set_loop_mode(gemm, assign_m_loop, CudaThreads(unit=cuda_warpgroup))
        sub_wg_m_loop = gemm.forward(assign_m_loop).body()[0]
        gemm = replace(gemm, sub_wg_m_loop, Sm90_mma_store_d_col_major_tf32(M=wg_M, N=wg_N))

    # Substitute cuda memset for 0-init.
    if enable_split_k:
        gemm = replace(gemm, gemm.find_loop("memset_batch"), cudaMemsetAsync0_3f32())

    # Specialize initial iteration of iter_k loop.
    # NB this doubles the size of the proc ... you can eliminate this
    # temporarily to make things easier to read.
    if True:
        gemm = cut_loop(gemm, iter_k_loop, 1)

    gemm = simplify(gemm)
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
