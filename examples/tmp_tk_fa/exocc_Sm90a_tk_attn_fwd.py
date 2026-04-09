from __future__ import annotations

from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.platforms.cuda_tk import *

from exo.scalars import bf16, f32, inf

from typing import List

import math
import time

cases: List[dict]
cases = []

T_type = bf16
L_type = f32

# Exo-GPU adaption of ThunderKittens mha_100.cu attention_forward.
def make_attn(Hdim: int, causal: bool, cases: List[dict]):
    assert Hdim in (64, 128)
    assert causal in (True, False)

    non_causal = not causal
    inv_sqrt_Hdim = Hdim ** -0.5
    log2_e = math.log2(math.e)
    py_ln_2 = math.log(2.0)
    num_consumers = 3
    qo_task_divisor = 64 * num_consumers
    RING = 256 // Hdim
    kv_height = 128
    SeqLen_divisor = kv_height

    my_warp_config = [
        CudaWarpConfig("consumer", 4 * num_consumers, setmaxnreg_inc=160),
        CudaWarpConfig("producer", 4, setmaxnreg_dec=32),
    ]

    o_tile_d = Sm90_TkRmemTileD(Hdim)
    att_tile = CudaTkWarpTile(16, kv_height)
    att_tile_d = Sm90_TkRmemTileD(kv_height)
    att_tile_a = Sm90_TkRmemTileA(kv_height)
    vec = att_tile_d.col_vec
    vec_layout = vec.layout
    assert vec.length == 16

    @proc
    def p(
        Batch: size, KV_Heads: size, Groups: size, SeqLen: size,
        O: T_type[Batch, KV_Heads, Groups, SeqLen, Hdim] @ CudaGmemLinear,
        lse: L_type[Batch, KV_Heads, Groups, SeqLen] @ CudaGmemLinear,
        Q: T_type[Batch, KV_Heads, Groups, SeqLen, Hdim] @ CudaGmemLinear,
        K: T_type[Batch, KV_Heads, SeqLen, Hdim] @ CudaGmemLinear,
        V: T_type[Batch, KV_Heads, SeqLen, Hdim] @ CudaGmemLinear,
    ):
      assert SeqLen % SeqLen_divisor == 0
      assert SeqLen > 0

      o_tm = O[:, :, :, :, :] @ Sm90_tensorMap(128, 1, 1, 1, 64, 64)
      q_tm = Q[:, :, :, :, :] @ Sm90_tensorMap(128, 1, 1, 1, 64, 64)
      k_tm = K[:, :, :, :] @ Sm90_tensorMap(128, 1, 1, kv_height, 64)
      v_tm = V[:, :, :, :] @ Sm90_tensorMap(128, 1, 1, kv_height, 64)
      lse_tm = lse[:, :, :, :] @ Sm90_tensorMap(0, 1, 1, 1, 64)

      with CudaDeviceFunction(warp_config=my_warp_config):
        for batch in cuda_tasks(0, Batch):
          for kv_head in cuda_tasks(0, KV_Heads):
            for group in cuda_tasks(0, Groups):
              for qo_task in cuda_tasks(0, (SeqLen + qo_task_divisor - 1) / qo_task_divisor):
                qo_smem: T_type[num_consumers, Hdim/64, 64, 64] @ Sm90_SmemSwizzled(128)
                k_smem: T_type[RING, Hdim/64, kv_height, 64] @ Sm90_SmemSwizzled(128)
                v_smem: T_type[RING, Hdim/64, kv_height, 64] @ Sm90_SmemSwizzled(128)
                lse_smem: L_type[num_consumers, 64] @ CudaSmemLinear

                q_produced: barrier[1 @ ring_buffer_by(1)] @ CudaMbarrierPreArrive(0)
                k_produced: barrier[(SeqLen / kv_height) @ ring_buffer_by(RING)] @ CudaMbarrierPreArrive(0)
                v_produced: barrier[(SeqLen / kv_height) @ ring_buffer_by(RING)] @ CudaMbarrierPreArrive(0)
                # ThunderKittens has a single compute_done variable.
                # These are separated to q/k/v here.
                # The q_consumed barrier only makes sense for a persistent kernel.
                q_consumed: barrier[2 @ ring_buffer_by(1)] @ CudaMbarrierPreArrive(1)
                k_consumed: barrier[(RING + SeqLen / kv_height) @ ring_buffer_by(RING)] @ CudaMbarrierPreArrive(RING)
                v_consumed: barrier[(RING + SeqLen / kv_height) @ ring_buffer_by(RING)] @ CudaMbarrierPreArrive(RING)

                with CudaWarps(0, 1, name="producer"):
                  # Load the Q tile for each consumer warpgroup.
                  Await(q_consumed[0], cuda_temporal, 0)
                  # TODO I should not have to unroll these tma_* loops.
                  for tma_consumer in seq(0, num_consumers):
                    for tma_hdim64 in seq(0, Hdim/64):
                      Sm90_tma_load_2d(
                        qo_smem[tma_consumer, tma_hdim64, :, :],
                        q_tm[batch, kv_head, group,
                             64 * (tma_consumer + qo_task * num_consumers) :
                             64 * (tma_consumer + qo_task * num_consumers) + 64,
                             64 * tma_hdim64 :
                             64 * tma_hdim64 + 64],
                        size0=64, size1=64, dst=T_type, src=T_type, smem_box=(1, 1, 1, 64, 64),
                      ) >> q_produced[0]
                  Arrive(cuda_temporal) >> q_produced[0]
                # End CudaWarps(0, 1, name="producer")

                # Wait for the Q tile to show up before entering main loop.
                with CudaWarps(name="consumer"):
                  Await(q_produced[0], cuda_generic_and_async_proxy, 0)
                  Arrive(cuda_in_order) >> q_consumed[1]
                # End CudaWarps(name="consumer")

                # Initialize consumer state
                att_block_d: f32[num_consumers, 4, 16, kv_height] @ att_tile_d
                att_block_scaled: f32[num_consumers, 4, 16, kv_height] @ att_tile
                att_block_exp2: f32[num_consumers, 4, 16, kv_height] @ att_tile
                att_block_a: T_type[num_consumers, 4, 16, kv_height] @ att_tile_a
                o_reg: f32[num_consumers, 4, 16, Hdim] @ o_tile_d
                max_vec: f32[num_consumers, 4, 16] @ vec
                norm_vec: f32[num_consumers, 4, 16] @ vec
                max_vec_last_scaled: f32[num_consumers, 4, 16] @ vec
                max_vec_last_exp2: f32[num_consumers, 4, 16] @ vec
                max_vec_scaled: f32[num_consumers, 4, 16] @ vec
                with CudaWarps(name="consumer"):
                  for consumer in cuda_threads(0, num_consumers, unit=cuda_warpgroup):
                    for w in cuda_threads(0, 4, unit=cuda_warp):
                      cuda_tk_vec_neg_inf(max_vec[consumer, w, :], length=16, dst=f32, layout=vec_layout)
                      cuda_tk_vec_zero(norm_vec[consumer, w, :], length=16, dst=f32, layout=vec_layout)
                      cuda_tk_tile_zero(o_reg[consumer, w, :, :], rows=16, cols=Hdim, dst=f32)

                # This handles both the 1/sqrt(d) and the base conversion for using exp2 rather than exp.
                scale: f32 @ CudaRmemUniform(512)
                scale = log2_e * inv_sqrt_Hdim

                # Accumulate along K/V height (row number increments by kv_height each iteration)
                for kv_idx in seq(0, SeqLen / kv_height):
                  # non-causal case must execute all iterations.
                  # causal case may early exit if we know the lower-left of the logit tile
                  # (row = (1 + qo_task) * num_consumers * 64 - 1, col = kv_idx * kv_height)
                  # is strictly above the main diagonal.
                  if non_causal or kv_idx * kv_height < (1 + qo_task) * num_consumers * 64:
                    with CudaWarps(0, 1, name="producer"):
                      # Load K tile for iteration, shared by all consumers.
                      Await(k_consumed[kv_idx], cuda_temporal, 0)
                      # TODO I should not have to unroll these tma_* loops.
                      for tma_hdim64 in seq(0, Hdim/64):
                        Sm90_tma_load_2d(
                          k_smem[kv_idx % RING, tma_hdim64, :, :],
                          k_tm[batch, kv_head,
                               kv_height * kv_idx :
                               kv_height * kv_idx + kv_height,
                               tma_hdim64 * 64 :
                               tma_hdim64 * 64 + 64],
                          size0=kv_height, size1=64, dst=T_type, src=T_type, smem_box=(1, 1, kv_height, 64),
                        ) >> k_produced[kv_idx]
                      Arrive(cuda_temporal) >> k_produced[kv_idx]
                      # Load V tile for iteration, shared by all consumers.
                      Await(v_consumed[kv_idx], cuda_temporal, 0)
                      for tma_hdim64 in seq(0, Hdim/64):
                        Sm90_tma_load_2d(
                          v_smem[kv_idx % RING, tma_hdim64, :, :],
                          v_tm[batch, kv_head,
                               kv_height * kv_idx :
                               kv_height * kv_idx + kv_height,
                               tma_hdim64 * 64 :
                               tma_hdim64 * 64 + 64],
                          size0=kv_height, size1=64, dst=T_type, src=T_type, smem_box=(1, 1, kv_height, 64),
                        ) >> v_produced[kv_idx]
                      Arrive(cuda_temporal) >> v_produced[kv_idx]
                    # End CudaWarps(0, 1, name="producer")

                    with CudaWarps(name="consumer"):
                      cg: barrier[num_consumers] @ Sm90_WgmmaCommitGroup
                      Await(k_produced[kv_idx], cuda_generic_and_async_proxy, 0)
                      for consumer in cuda_threads(0, num_consumers, unit=cuda_warpgroup):
                        # First MMA writes block QKt to att_block_d.
                        # We don't accumulate across kv-iterations, so use scale_d=0 to reset.
                        # head-dim is the K dimension, and we accum in 2 steps if Hdim is 128.
                        Fence(wgmma_fence_1, wgmma_fence_2)
                        Sm90_tk_zero_scale_d(att_block_d[consumer, :, :, :], D=f32, N=kv_height)
                        for hdim64 in seq(0, Hdim / 64, pragma_unroll=0):
                          Sm90_tk_mma_row_col(
                            att_block_d[consumer, :, :, :],
                            qo_smem[consumer, hdim64, :, :],
                            k_smem[kv_idx % RING, hdim64, :, :],
                            D=f32, A=T_type, B=T_type, N=kv_height, K=64,
                          )

                        # Compute max_vec_last_scaled while we wait for wgmma to retire.
                        Arrive(wgmma_async) >> cg[consumer]
                        for w in cuda_threads(0, 4, unit=cuda_warp):
                          cuda_tk_vec_mul_3op_scalar(
                            max_vec_last_scaled[consumer, w, :], max_vec[consumer, w, :], scale,
                            dst=f32, lhs=f32, rhs=f32, length=16, layout=vec_layout)
                        Await(cg[consumer], cuda_generic_and_async_proxy, 0)

                      Arrive(cuda_in_order) >> k_consumed[kv_idx + RING]
                      for consumer in cuda_threads(0, num_consumers, unit=cuda_warpgroup):
                        for w in cuda_threads(0, 4, unit=cuda_warp):
                          # Each warp updates its own [16, kv_height] tiles with non-async code.
                          if causal:
                            cuda_tk_make_causal_neg_inf(
                              # current row offset
                              64 * (qo_task * num_consumers + consumer) + 16 * w,
                              # current col offset
                              kv_idx * kv_height,
                              att_block_d[consumer, w, :, :],
                              dst=f32, rows=16, cols=kv_height,
                            )
                          # End if causal
                          cuda_tk_row_max(
                            max_vec[consumer, w, :], att_block_d[consumer, w, :, :],
                            dst=f32, src=f32, rows=16, cols=kv_height)
                          cuda_tk_tile_mul_3op_scalar(
                            att_block_scaled[consumer, w, :, :], att_block_d[consumer, w, :, :], scale,
                            dst=f32, lhs=f32, rhs=f32, rows=16, cols=kv_height)
                          cuda_tk_vec_mul_3op_scalar(
                            max_vec_scaled[consumer, w, :], max_vec[consumer, w, :], scale,
                            dst=f32, lhs=f32, rhs=f32, length=16, layout=vec_layout)
                          cuda_tk_sub_row(
                            att_block_scaled[consumer, w, :, :], max_vec_scaled[consumer, w, :],
                            dst=f32, src=f32, rows=16, cols=kv_height)
                          cuda_tk_tile_exp2(
                            att_block_exp2[consumer, w, :, :], att_block_scaled[consumer, w, :, :],
                            dst=f32, src=f32, rows=16, cols=kv_height)
                          cuda_tk_vec_sub_lhs(
                            max_vec_last_scaled[consumer, w, :], max_vec_scaled[consumer, w, :],
                            dst=f32, src=f32, length=16, layout=vec_layout)
                          cuda_tk_vec_exp2(
                            max_vec_last_exp2[consumer, w, :], max_vec_last_scaled[consumer, w, :],
                            dst=f32, src=f32, length=16, layout=vec_layout)
                          cuda_tk_vec_mul_lhs(
                            norm_vec[consumer, w, :], max_vec_last_exp2[consumer, w, :],
                            dst=f32, src=f32, length=16, layout=vec_layout)
                          cuda_tk_row_sum(
                            norm_vec[consumer, w, :], att_block_exp2[consumer, w, :, :],
                            dst=f32, src=f32, rows=16, cols=kv_height)
                          rmem_zero: f32 @ CudaRmemUniform(32)
                          rmem_zero = 0
                          cuda_tk_tile_add_lhs_scalar(
                            att_block_exp2[consumer, w, :, :], rmem_zero,
                            dst=f32, src=f32, rows=16, cols=kv_height)
                          cuda_tk_tile_copy(
                            att_block_a[consumer, w, :, :], att_block_exp2[consumer, w, :, :],
                            dst=T_type, src=f32, rows=16, cols=kv_height)
                          cuda_tk_mul_row(
                            o_reg[consumer, w, :, :], max_vec_last_exp2[consumer, w, :],
                            dst=f32, src=f32, rows=16, cols=Hdim)
                        # End for w in cuda_threads(0, 4, unit=cuda_warp)

                      # Second MMA accumulates to O the product of att_block_exp2
                      # (cast to T_type, att_block_a) and the current tile of V.
                      # Both are row major now.
                      Await(v_produced[kv_idx], cuda_generic_and_async_proxy, 0)
                      for consumer in cuda_threads(0, num_consumers, unit=cuda_warpgroup):
                        Fence(wgmma_fence_1, wgmma_fence_2)
                        Sm90_tk_mma_rmem_row(
                          o_reg[consumer, :, :, :],
                          att_block_a[consumer, :, :, :],
                          v_smem[kv_idx % RING, :, :, :],
                          D=f32, A=T_type, B=T_type, N64=Hdim // 64, K=kv_height,
                        )
                        Arrive(wgmma_async) >> cg[consumer]
                        Await(cg[consumer], cuda_generic_and_async_proxy, 0)
                      Arrive(cuda_in_order) >> v_consumed[kv_idx + RING]
                    # End with CudaWarps(name="consumer")
                  # End causal thing
                # End for kv_idx

                # Epilogue 1/2: each consumer writes out its own output tile and lse_vec
                # to SMEM after some final scaling.
                with CudaWarps(name="consumer"):
                  for consumer in cuda_threads(0, num_consumers, unit=cuda_warpgroup):
                    for w in cuda_threads(0, 4, unit=cuda_warp):
                      cuda_tk_div_row(
                        o_reg[consumer, w, :, :], norm_vec[consumer, w, :],
                        dst=f32, src=f32, rows=16, cols=Hdim)
                      cuda_tk_store_rs_inner_cols_64(
                        qo_smem[consumer, :, w * 16: w * 16 + 16, :],
                        o_reg[consumer, w, :, :],
                        dst=T_type, src=f32, rows=16, outer_cols=Hdim // 64,
                      )
                      ln_2: f32 @ CudaRmemUniform(32)
                      ln_2 = py_ln_2
                      cuda_tk_vec_mul_lhs_scalar(
                        max_vec_scaled[consumer, w, :], ln_2,
                        dst=f32, src=f32, length=16, layout=vec_layout)
                      norm_vec_log: f32[16] @ vec
                      cuda_tk_vec_log(
                        norm_vec_log[:], norm_vec[consumer, w, :],
                        dst=f32, src=f32, length=16, layout=vec_layout)
                      cuda_tk_vec_add_reduce(
                        norm_vec_log[:], max_vec_scaled[consumer, w, :],
                        dst=f32, src=f32, length=16, layout=vec_layout)
                      # NOTE: ThunderKittens here additionally scales norm_vec_log by -sqrt(Hdim)
                      cuda_tk_store_vec_rs(
                        lse_smem[consumer, w * 16 : w * 16 + 16], norm_vec_log[:],
                        dst=L_type, src=f32, length=16, layout=vec_layout)

                Fence(cuda_in_order, cuda_generic_and_async_proxy)

                # Epilogue 2/2: copy staged SMEM outputs to GMEM
                with CudaWarps(name="consumer"):
                  for consumer in cuda_threads(0, num_consumers, unit=cuda_warpgroup):
                    with CudaWarps(0, 1):
                      cg: barrier @ Sm90_TmaCommitGroup
                      for hdim64 in seq(0, Hdim / 64):
                        Sm90_tma_store_2d(
                          o_tm[batch, kv_head, group,
                               64 * (consumer + qo_task * num_consumers) :
                               64 * (consumer + qo_task * num_consumers) + 64,
                               64 * hdim64 :
                               64 * hdim64 + 64,
                          ],
                          qo_smem[consumer, hdim64, :, :],
                          dst=T_type, src=T_type, size0=64, size1=64, smem_box=(1, 1, 1, 64, 64), swizzle=128,
                        )
                      Sm90_tma_store_1d(
                        lse_tm[batch, kv_head, group,
                               64 * (consumer + qo_task * num_consumers) :
                               64 * (consumer + qo_task * num_consumers) + 64,
                        ],
                        lse_smem[consumer, :],
                        dst=L_type, src=L_type, size0=64, smem_box=(1, 1, 1, 64), swizzle=0,
                      )
                      Arrive(tma_to_gmem_async) >> cg
                      Await(cg, cuda_in_order, 0)

                Fence(cuda_in_order, cuda_in_order)

    p = simplify(p)
    p = rename(p, f"exo_tk_attn_fwd_Hdim{Hdim}" + "_causal" * causal)
    for loop_c in p.find_all("for tma_consumer in _:_"):
        p = unroll_loop(p, loop_c)
    for loop_c in p.find_all("for tma_hdim64 in _:_"):
        p = unroll_loop(p, loop_c)

    sync_check_before = time.time()
    p.sync_check(Batch=1, KV_Heads=2, Groups=2, SeqLen=640)
    dt = time.time() - sync_check_before
    print(f"{p.name()}.sync_check: %.0f ms" % (1000 * dt,))

    if cases is not None:
        j_case = {
            "algorithm": "attn_fwd",
            "T_type": str(T_type),
            "L_type": str(L_type),
            "proc": p.name(),
            "args": ["Batch", "KV_Heads", "Groups", "SeqLen", "O", "lse", "Q", "K", "V"],
            "SeqLen_divisor": SeqLen_divisor,
            "Hdim": Hdim,
            "causal": causal,
        }
        cases.append(j_case)

    return p


# attn_64 = make_attn(64, False, cases)
attn_128 = make_attn(128, False, cases)
# attn_64_causal = make_attn(64, True, cases)
# attn_128_causal = make_attn(128, True, cases)


import json
json.dump(cases, open(__file__ + ".json", "w"))
