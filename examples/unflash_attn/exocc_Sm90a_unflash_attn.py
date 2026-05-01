from __future__ import annotations

from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.platforms.cuda_tk import *
from exo.platforms.Sm90.tk_gemm_util import GemmConfig, handwrite_gemm

from exo.scalars import bf16, f32, inf

from typing import List

cases: List[dict]
cases = []

T_type = bf16
L_type = f32

def enable_gemm_window(p):
    for nm in "ABC":
        c = p.find_alloc_or_arg(nm)
        p = set_window(p, c)
    return p

QKt_gemm = enable_gemm_window(handwrite_gemm(GemmConfig(
    cta_M=128,
    cta_N=64,
    A_type=T_type,
    B_type=T_type,
    C_type=f32,
    D_type=f32,
    A_major="row",
    B_major="col",
    C_major="row",
)))

SO_gemm = enable_gemm_window(handwrite_gemm(GemmConfig(
    cta_M=128,
    cta_N=64,
    A_type=T_type,
    B_type=T_type,
    C_type=T_type,
    D_type=f32,
    A_major="row",
    B_major="row",
    C_major="row",
)))

@proc
def S_kernel(
    causal: bool,
    SeqLen: size,
    scale_factor: f32 @ CudaGridConstant,
    lse: [L_type][SeqLen] @ CudaGmemLinear,
    S: [T_type][SeqLen, SeqLen] @ CudaGmemLinear,
    QKt: [f32][SeqLen, SeqLen] @ CudaGmemLinear,
):
    assert SeqLen % 16 == 0
    assert stride(lse, 0) == 1
    assert stride(S, 1) == 1
    assert stride(QKt, 1) == 1

    with CudaDeviceFunction(blockDim=32, blocks_per_sm=32):
        for r in cuda_tasks(0, SeqLen / 16):
            tile: f32[16, 16] @ CudaTkWarpTile(16, 16)
            max_accum: f32[16] @ CudaTkWarpTile(16, 16).col_vec
            sum_accum: f32[16] @ CudaTkWarpTile(16, 16).col_vec
            cuda_tk_vec_zero(sum_accum[:], dst=f32, length=16, layout="ortho")
            cuda_tk_vec_neg_inf(max_accum[:], dst=f32, length=16, layout="ortho")
            # Find row maximum
            for c in seq(0, SeqLen / 16):
                cuda_tk_load_rg(
                    tile[:, :],
                    QKt[16 * r : 16 * r + 16, 16 * c : 16 * c + 16],
                    dst=f32, src=f32, size0=16, size1=16)
                cuda_tk_tile_mul_lhs_scalar(tile[:, :], scale_factor, dst=f32, src=f32, rows=16, cols=16)
                if causal:
                    cuda_tk_make_causal_neg_inf(16 * r, 16 * c, tile[:, :], dst=f32, rows=16, cols=16)
                cuda_tk_row_max(max_accum[:], tile[:, :], dst=f32, src=f32, rows=16, cols=16)
            # Sum of exp of each row.
            for c in seq(0, SeqLen / 16):
                exp_tile: f32[16, 16] @ CudaTkWarpTile(16, 16)
                cuda_tk_load_rg(
                    tile[:, :],
                    QKt[16 * r : 16 * r + 16, 16 * c : 16 * c + 16],
                    dst=f32, src=f32, size0=16, size1=16)
                cuda_tk_tile_mul_lhs_scalar(tile[:, :], scale_factor, dst=f32, src=f32, rows=16, cols=16)
                if causal:
                    cuda_tk_make_causal_neg_inf(16 * r, 16 * c, tile[:, :], dst=f32, rows=16, cols=16)
                cuda_tk_sub_row(tile[:, :], max_accum[:], dst=f32, src=f32, rows=16, cols=16)
                cuda_tk_tile_exp(exp_tile[:, :], tile[:, :], dst=f32, src=f32, rows=16, cols=16)
                cuda_tk_row_sum(sum_accum[:], exp_tile[:, :], dst=f32, src=f32, rows=16, cols=16)
            # Write out each tile exp, divided by denominator
            # Also accumulate row-sum of S[r, c] prior to exp.
            rcp_tile: f32[16, 16] @ CudaTkWarpTile(16, 16)
            cuda_tk_tile_one(rcp_tile[:, :], dst=f32, rows=16, cols=16)
            cuda_tk_div_row(rcp_tile[:, :], sum_accum[:], dst=f32, src=f32, rows=16, cols=16)
            se_accum: f32[16] @ CudaTkWarpTile(16, 16).col_vec
            cuda_tk_vec_zero(se_accum[:], dst=f32, length=16, layout="ortho")
            for c in seq(0, SeqLen / 16):
                exp_tile: f32[16, 16] @ CudaTkWarpTile(16, 16)
                cuda_tk_load_rg(
                    tile[:, :],
                    QKt[16 * r : 16 * r + 16, 16 * c : 16 * c + 16],
                    dst=f32, src=f32, size0=16, size1=16)
                cuda_tk_tile_mul_lhs_scalar(tile[:, :], scale_factor, dst=f32, src=f32, rows=16, cols=16)
                if causal:
                    cuda_tk_make_causal_neg_inf(16 * r, 16 * c, tile[:, :], dst=f32, rows=16, cols=16)
                cuda_tk_sub_row(tile[:, :], max_accum[:], dst=f32, src=f32, rows=16, cols=16)
                cuda_tk_tile_exp(exp_tile[:, :], tile[:, :], dst=f32, src=f32, rows=16, cols=16)
                cuda_tk_row_sum(se_accum[:], exp_tile[:, :], dst=f32, src=f32, rows=16, cols=16)
                cuda_tk_tile_mul_lhs(exp_tile[:, :], rcp_tile[:, :], dst=f32, src=f32, rows=16, cols=16)
                cuda_tk_store_rg(
                    S[16 * r : 16 * r + 16, 16 * c : 16 * c + 16],
                    exp_tile[:, :],
                    dst=T_type, src=f32, size0=16, size1=16)
            # Write out log-sum-exp
            l_smem: L_type[16] @ CudaSmemLinear
            l_rmem: f32[16] @ CudaTkWarpTile(16, 16).col_vec
            cuda_tk_vec_log(l_rmem[:], se_accum[:], length=16, layout="ortho", dst=f32, src=f32)
            cuda_tk_vec_add_reduce(l_rmem[:], max_accum[:], length=16, layout="ortho", dst=f32, src=f32)
            cuda_tk_store_vec_rs(l_smem[:], l_rmem[:], length=16, layout="ortho", dst=L_type, src=f32)
            Fence(cuda_in_order, cuda_in_order)
            for t in cuda_threads(0, 16):
                lse[r * 16 + t] = l_smem[t]
            Fence(cuda_in_order, cuda_in_order)

S_kernel = simplify(S_kernel)
S_kernel = rename(S_kernel, "unflash_attn_S_kernel")
S_kernel.sync_check(SeqLen=768)

smoke_test = False

@proc
def smoke_test_overwrite(
        Batch: size, KV_Heads: size, Groups: size, SeqLen: size, Hdim: size,
        O: [T_type][Batch, KV_Heads, Groups, SeqLen, Hdim] @ CudaGmemLinear,
        lse: [L_type][Batch, KV_Heads, Groups, SeqLen] @ CudaGmemLinear,
):
    if Batch > 1:
        if KV_Heads > 19:
            if Groups > 1:
                if SeqLen > 1000:
                    if Hdim > 32:
                        with CudaDeviceFunction(blockDim=32):
                            for task in cuda_tasks(0, 1):
                                for tid in cuda_threads(0, 1):
                                    O[1, 19, 1, 1000, 32] = 1337
                                    lse[1, 19, 1, 1000] = 1337


def make_attn(Hdim: int, causal: bool, cases: List[dict]):
    assert Hdim in (64, 128)
    assert causal in (True, False)

    py_scale_factor = Hdim ** -0.5

    @proc
    def p(
        Batch: size, KV_Heads: size, Groups: size, SeqLen: size,
        O: T_type[Batch, KV_Heads, Groups, SeqLen, Hdim] @ CudaGmemLinear,
        lse: L_type[Batch, KV_Heads, Groups, SeqLen] @ CudaGmemLinear,
        Q: T_type[Batch, KV_Heads, Groups, SeqLen, 1, Hdim] @ CudaGmemLinear,
        K: T_type[Batch, KV_Heads, SeqLen, 1, Hdim] @ CudaGmemLinear,
        V: T_type[Batch, KV_Heads, 1, SeqLen, Hdim] @ CudaGmemLinear,
    ):
        assert SeqLen % 256 == 0
        # The 1's in the tensor sizes is to fill the K_split dimension
        # (Exo cannot redimensionalize).
        QKt: f32[1, SeqLen, SeqLen] @ CudaGmemLinear
        S: T_type[1, SeqLen, 1, SeqLen] @ CudaGmemLinear
        scale_factor: f32 @ CudaGridConstant
        scale_factor = py_scale_factor

        for batch in seq(0, Batch):
            for kv_head in seq(0, KV_Heads):
                for group in seq(0, Groups):
                    QKt_gemm(
                        1,        # L
                        SeqLen,   # M
                        SeqLen,   # N
                        1,        # K_split
                        Hdim,     # K_cluster
                        Q[batch:batch+1, kv_head, group, :, :, :],
                        K[batch:batch+1, kv_head, :, :, :],
                        QKt[:, :, :],
                    )
                    S_kernel(
                        causal,
                        SeqLen,
                        scale_factor,
                        lse[batch, kv_head, group, :],
                        S[0, :, 0, :],
                        QKt[0, :, :],
                    )
                    SO_gemm(
                        1,        # L
                        SeqLen,   # M
                        Hdim,     # N
                        1,        # K_split
                        SeqLen,   # K_cluster
                        S[:, :, :, :],
                        V[batch:batch+1, kv_head, :, :, :],
                        O[batch:batch+1, kv_head, group, :, :],
                    )
        if smoke_test:
            smoke_test_overwrite(Batch, KV_Heads, Groups, SeqLen, Hdim, O[:, :, :, :, :], lse[:, :, :, :])

    p = simplify(p)
    p = rename(p, f"unflash_attn_Hdim{Hdim}" + "_causal" * causal)

    if cases is not None:
        j_case = {
            "algorithm": "attn_fwd",
            "T_type": str(T_type),
            "L_type": str(L_type),
            "proc": p.name(),
            "args": ["Batch", "KV_Heads", "Groups", "SeqLen", "O", "lse", "Q", "K", "V"],
            "SeqLen_divisor": 256,
            "Hdim": Hdim,
            "causal": causal,
            "test_correctness_only": True,
        }
        cases.append(j_case)

    return p


unflash_attn_64 = make_attn(64, False, cases)
unflash_attn_128 = make_attn(128, False, cases)
unflash_attn_64_causal = make_attn(64, True, cases)
unflash_attn_128_causal = make_attn(128, True, cases)


import json
json.dump(cases, open(__file__ + ".json", "w"))
