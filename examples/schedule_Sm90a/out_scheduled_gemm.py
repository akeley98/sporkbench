def scheduled_gemm(L : size, M : size, N : size, K_split : size, cluster_K : size, A : f32[L, M, K_split, cluster_K] @CudaGmemLinear, B : f32[L, N, K_split, cluster_K] @CudaGmemLinear, C : f32[L, N, M] @CudaGmemLinear):
  assert L > 0
  assert M > 0
  assert N > 0
  assert cluster_K > 0
  assert cluster_K % 4 == 0
  A_tensorMap = A[0:L, 0:M, 0:K_split, 0:cluster_K] @ Sm90_tensorMap(128, 1, 64, 1, 32)
  B_tensorMap = B[0:L, 0:N, 0:K_split, 0:cluster_K] @ Sm90_tensorMap(128, 1, 128, 1, 32)
  with CudaDeviceFunction(clusterDim=4, warp_config=[CudaWarpConfig(name='producer', count=1, setmaxnreg_dec=40, setmaxnreg_inc=None), CudaWarpConfig(name='unused', count=3, setmaxnreg_dec=40, setmaxnreg_inc=None), CudaWarpConfig(name='consumer', count=8, setmaxnreg_dec=None, setmaxnreg_inc=232)]):
    for batch in cuda_tasks(0, L):
      for task_k in cuda_tasks(0, K_split):
        for task_n in cuda_tasks(0, (511 + N) / 512):
          for task_m in cuda_tasks(0, (255 + M) / 256):
            D_rmem : f32[2, 2, 2, 64, 256] @Sm90_RmemMatrixD(64, 256)
            for cta_m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
              for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                with CudaWarps(name='consumer'):
                  for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                    for sub_wg_m in seq(0, 64):
                      if sub_wg_m + 64 * wg_m + 128 * cta_m + 256 * task_m < M:
                        for sub_cta_n in seq(0, 256):
                          if sub_cta_n + 256 * cta_n + 512 * task_n < N:
                            D_rmem[cta_m, cta_n, wg_m, sub_wg_m, sub_cta_n] = 0
            A_smem : f32[2, 2, 4, 16, 8, 32] @Sm90_SmemSwizzled(128,)
            B_smem : f32[2, 2, 4, 32, 8, 32] @Sm90_SmemSwizzled(128,)
            for iter_k in seq(0, 1):
              with CudaWarps(name='producer'):
                for cta_m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
                  for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    PLACEHOLDER_WAR_AWAIT()
                  for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    for i0 in seq(0, 128):
                      for i1 in seq(0, 32):
                        if i0 + 128 * cta_m + 256 * task_m < M and i1 + 32 * iter_k < cluster_K:
                          A_smem[cta_m, cta_n, iter_k, i0 / 8, i0 % 8, i1] = A_tensorMap[batch, i0 + 128 * cta_m + 256 * task_m, task_k, i1 + 32 * iter_k]
                for cta_n in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster_strided(2)):
                  for cta_m in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    for i0 in seq(0, 256):
                      for i1 in seq(0, 32):
                        if i0 + 256 * cta_n + 512 * task_n < N and i1 + 32 * iter_k < cluster_K:
                          B_smem[cta_m, cta_n, iter_k, i0 / 8, i0 % 8, i1] = B_tensorMap[batch, i0 + 256 * cta_n + 512 * task_n, task_k, i1 + 32 * iter_k]
                  for cta_m in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    PLACEHOLDER_RAW_ARRIVE()
              with CudaWarps(name='consumer'):
                for cta_m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
                  for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    PLACEHOLDER_RAW_AWAIT()
                    for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                      Fence(wgmma_fence_1, wgmma_fence_2)  # Fence_31576
                      for mma_k in seq(0, 4):
                        for sub_wg_m in seq(0, 64):
                          if sub_wg_m + 64 * wg_m + 128 * cta_m + 256 * task_m < M:
                            for sub_cta_n in seq(0, 256):
                              if sub_cta_n + 256 * cta_n + 512 * task_n < N:
                                for sub_mma_k in seq(0, 8):
                                  if sub_mma_k + 8 * mma_k + 32 * iter_k < cluster_K:
                                    D_rmem[cta_m, cta_n, wg_m, sub_wg_m, sub_cta_n] += A_smem[cta_m, cta_n, iter_k, (sub_wg_m + 64 * wg_m) / 8, sub_wg_m % 8, sub_mma_k + 8 * mma_k] * B_smem[cta_m, cta_n, iter_k, sub_cta_n / 8, sub_cta_n % 8, sub_mma_k + 8 * mma_k]
                      PLACEHOLDER_CG_ARRIVE()
                    PLACEHOLDER_WAR_ARRIVE()
            for iter_k in seq(1, (31 + cluster_K) / 32):
              with CudaWarps(name='producer'):
                for cta_m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
                  for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    PLACEHOLDER_WAR_AWAIT()
                  for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    for i0 in seq(0, 128):
                      for i1 in seq(0, 32):
                        if i0 + 128 * cta_m + 256 * task_m < M and i1 + 32 * iter_k < cluster_K:
                          A_smem[cta_m, cta_n, iter_k % 4, i0 / 8, i0 % 8, i1] = A_tensorMap[batch, i0 + 128 * cta_m + 256 * task_m, task_k, i1 + 32 * iter_k]
                for cta_n in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster_strided(2)):
                  for cta_m in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    for i0 in seq(0, 256):
                      for i1 in seq(0, 32):
                        if i0 + 256 * cta_n + 512 * task_n < N and i1 + 32 * iter_k < cluster_K:
                          B_smem[cta_m, cta_n, iter_k % 4, i0 / 8, i0 % 8, i1] = B_tensorMap[batch, i0 + 256 * cta_n + 512 * task_n, task_k, i1 + 32 * iter_k]
                  for cta_m in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    PLACEHOLDER_RAW_ARRIVE()
              with CudaWarps(name='consumer'):
                for cta_m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
                  for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    PLACEHOLDER_RAW_AWAIT()
                    for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                      Fence(wgmma_fence_1, wgmma_fence_2)  # Fence_33913
                      for mma_k in seq(0, 4):
                        for sub_wg_m in seq(0, 64):
                          if sub_wg_m + 64 * wg_m + 128 * cta_m + 256 * task_m < M:
                            for sub_cta_n in seq(0, 256):
                              if sub_cta_n + 256 * cta_n + 512 * task_n < N:
                                for sub_mma_k in seq(0, 8):
                                  if sub_mma_k + 8 * mma_k + 32 * iter_k < cluster_K:
                                    D_rmem[cta_m, cta_n, wg_m, sub_wg_m, sub_cta_n] += A_smem[cta_m, cta_n, iter_k % 4, (sub_wg_m + 64 * wg_m) / 8, sub_wg_m % 8, sub_mma_k + 8 * mma_k] * B_smem[cta_m, cta_n, iter_k % 4, sub_cta_n / 8, sub_cta_n % 8, sub_mma_k + 8 * mma_k]
                      PLACEHOLDER_CG_ARRIVE()
                    PLACEHOLDER_WAR_ARRIVE()
            for cta_m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
              for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                with CudaWarps(name='consumer'):
                  for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                    PLACEHOLDER_CG_AWAIT()
            for cta_m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
              for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                with CudaWarps(name='consumer'):
                  for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                    Sm90_mma_store_d_col_major_tf32(-(256 * task_m) - 128 * cta_m - 64 * wg_m + M, -(512 * task_n) - 256 * cta_n + N, C[batch, 256 * cta_n + 512 * task_n:256 + 256 * cta_n + 512 * task_n, 64 * wg_m + 128 * cta_m + 256 * task_m:64 + 64 * wg_m + 128 * cta_m + 256 * task_m], D_rmem[cta_m, cta_n, wg_m, 0:64, 0:256], M=64, N=256)
            Fence(cuda_in_order, cuda_in_order)  # Fence_33163