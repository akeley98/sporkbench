from dataclasses import dataclass

# Usage:
#
# from Sm80_gemm_pre_config import config
# config.smem_M = ...
# config.smem_N = ...
#
# # Generates kernels based on above config and places it in this module.
# from Sm80_gen_gemm import *
#
# # Autogenerate json file
# import json
# json.dump(cases, open(__file__ + ".json", "w"))

@dataclass(slots=True)
class Sm80GemmConfig:
    smem_M: int = None
    smem_N: int = None
    smem_K: int = 16
    warp_M: int = None
    warp_N: int = None
    enable_split_k: bool = None
    blocks_per_sm: int = None
    mbarrier_ring: int = 3
    mbarrier_lag: int = 1

    def make_proc_name(self, sync_name: str):
        suffix = "dataParallel" if not self.enable_split_k else "splitK"
        return f"xgemm_Sm80_m{self.smem_M}n{self.smem_N}k{self.smem_K}_m{self.warp_M}n{self.warp_N}_{sync_name}_{suffix}"

config = Sm80GemmConfig()
