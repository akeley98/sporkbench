from template.make_Sm90a_gemm import make_Sm90a_gemm, Sm90aGemmConfig
from gemv_for_benchmark import gemv_warp_coop_8

from typing import List, Tuple
import json
import timeit
import os
import sys

# problem_sizes = [512, 680, 856]
problem_sizes = [512, 680, 856, 1024, 1536, 2048, 3072, 4096]


non_split_k_config = Sm90aGemmConfig(128, 256, tma_to_gmem=False, enable_split_k=False)
split_k_config = Sm90aGemmConfig(128, 256, tma_to_gmem=True, enable_split_k=True)
non_split_k_gemm = make_Sm90a_gemm(non_split_k_config, 2, 1)
split_k_gemm = make_Sm90a_gemm(split_k_config, 2, 1)
K_split = 4


def sz_to_non_split_k_gemm_args(sz):
    return dict(L=1, M=sz, N=sz, cluster_K=sz, K_split=1)


def sz_to_split_k_gemm_args(sz):
    return dict(L=1, M=sz, N=sz, cluster_K=sz // K_split, K_split=K_split)


def sz_to_gemv_args(sz):
    return dict(M=sz, K=sz)


def benchmark(p, sz_to_args) -> List[Tuple[int, float]]:
    results = []
    for number in (1, 5):  # 1 is warmup
        for sz in problem_sizes:
            def callback():
                p.sync_check(**sz_to_args(sz))
            dt = timeit.timeit(callback, number=number) / number
            print("%4d %9.2f %s" % (sz, dt, p.name()), file=sys.stderr)
            if number > 1:
                results.append((sz, dt))
    return results


if __name__ == "__main__":
    results = {}

    results["non-split-k sm_90a gemm"] = benchmark(non_split_k_gemm, sz_to_non_split_k_gemm_args)
    results[f"split-k sm_90a gemm (K_split={K_split})"] = benchmark(split_k_gemm, sz_to_split_k_gemm_args)
    results["gemv"] = benchmark(gemv_warp_coop_8, sz_to_gemv_args)

    this_dir = os.path.split(__file__)[0]
    out_name = os.path.join(this_dir, "benchmark_sync_check.json")
    print(out_name)
    with open(out_name, "w") as f:
        json.dump(results, f)
