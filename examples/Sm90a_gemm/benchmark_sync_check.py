from exocc_Sm90a_m128n256_rmemC import *
from typing import List, Tuple
import json
import timeit
import os

# problem_sizes = [512]
problem_sizes = [512, 1024, 1536, 2048, 3072, 4096]

def benchmark(p, number) -> List[Tuple[int, float]]:
    results = []
    for sz in problem_sizes:
        def callback():
            p.sync_check(L=1, M=sz, N=sz, cluster_K=sz, K_split=1)
        dt = timeit.timeit(callback, number=number)
        print(sz, dt)
        results.append((sz, dt))
    return results

if __name__ == "__main__":
    benchmark(gemm_m2n2, 1)  # Warmup
    results = benchmark(gemm_m2n2, 5)

    this_dir = os.path.split(__file__)[0]
    out_name = os.path.join(this_dir, "benchmark_sync_check.json")
    print(out_name)
    with open(out_name, "w") as f:
        json.dump(results, f)
