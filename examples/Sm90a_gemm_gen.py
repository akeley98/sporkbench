from Sm90a_gemm_pre_config import config

assert isinstance(config.py_filename, str), "Need to import Sm90a_gemm_pre_config first and set config variables"
smem_M = int(config.smem_M)
smem_N = int(config.smem_N)
RING = int(config.RING)

