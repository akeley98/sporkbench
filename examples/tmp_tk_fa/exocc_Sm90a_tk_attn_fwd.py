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

cases: List[dict]
cases = []

T_type = bf16
L_type = f32

# Exo-GPU translation of ThunderKittens mha_100.cu attention_forward.
def make_attn(Hdim: int, causal: bool, cases: List[dict]):
    assert Hdim in (64, 128)
    assert causal in (True, False)

    py_scale_factor = Hdim ** -0.5
    log2_e = math.log2(math.e)
    ln_2 = math.log(2.0)
    SeqLen_divisor = 16
    num_consumers = 3
    qo_task_divisor = 64 * num_consumers

    my_warp_config = [
        CudaWarpConfig("consumer", 4 * num_consumers, setmaxnerg_inc=160),
        CudaWarpConfig("producer", 4, setmaxnreg_dec=32),
    ]

    @proc
    def p(
        Batch: size, KV_Heads: size, Groups: size, SeqLen: size,
        O: T_type[Batch, KV_Heads, Groups, SeqLen, Hdim] @ CudaGmemLinear,
        l_vec: L_type[Batch, KV_Heads, Groups, SeqLen] @ CudaGmemLinear,
        Q: T_type[Batch, KV_Heads, Groups, SeqLen, Hdim] @ CudaGmemLinear,
        K: T_type[Batch, KV_Heads, SeqLen, Hdim] @ CudaGmemLinear,
        V: T_type[Batch, KV_Heads, SeqLen, Hdim] @ CudaGmemLinear,
    ):
        assert SeqLen % SeqLen_divisor == 0

        with CudaDeviceFunction(warp_config=my_warp_config):
          for batch in cuda_tasks(0, Batch):
            for kv_head in cuda_tasks(0, KV_Heads):
              for group in cuda_tasks(0, Group):
                for qo_task in cuda_tasks(0, (SeqLen + qo_task_divisor - 1) / qo_task_divisor):
                  qo_smem: T_type[num_consumers, Hdim/64, 64, 64] @ Sm90_SmemSwizzled(128)
                  k_smem: T_type[num_consumers, RING, Hdim/64, 64, 64] @ Sm90_SmemSwizzled(128)
                  v_smem: T_type[num_consumers, RING, Hdim/64, 64, 64] @ Sm90_SmemSwizzled(128)
                  l_smem: L_type[num_consumers, 64, Hdim] @ CudaSmemLinear

    p = simplify(p)
    p = rename(p, f"exo_tk_attn_fwd_Hdim{Hdim}" + "_causal" * causal)

    if cases is not None:
        j_case = {
            "algorithm": "attn_fwd",
            "T_type": str(T_type),
            "L_type": str(L_type),
            "proc": p.name(),
            "args": ["Batch", "KV_Heads", "Groups", "SeqLen", "O", "l_vec", "Q", "K", "V"],
            "SeqLen_divisor": SeqLen_divisor,
            "Hdim": Hdim,
            "causal": causal,
        }
        cases.append(j_case)

    return p
