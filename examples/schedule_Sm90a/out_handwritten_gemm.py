def handwritten_gemm(L : size, M : size, N : size, K_split : size, cluster_K : size, A : f32[L, M, K_split, cluster_K] @CudaGmemLinear, B : f32[L, N, K_split, cluster_K] @CudaGmemLinear, C : f32[L, N, M] @CudaGmemLinear):
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
            raw : barrier[2, 2] @CudaMbarrier
            war : barrier(raw)[2, 2] @CudaMbarrier
            cg : barrier[2, 2, 2] @CudaCommitGroup
            D_rmem : f32[2, 2, 2, 64, 256] @Sm90_RmemMatrixD(64, 256)
            for cta_m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
              for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                with CudaWarps(name='consumer'):
                  for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                    Sm90_zero_scale_d_f32(D_rmem[cta_m, cta_n, wg_m, 0:64, 0:256], M=64, N=256)
            A_smem : f32[2, 2, 4, 16, 8, 32] @Sm90_SmemSwizzled(128,)
            B_smem : f32[2, 2, 4, 32, 8, 32] @Sm90_SmemSwizzled(128,)
            for iter_k in seq(0, 1):
              with CudaWarps(name='producer'):
                for cta_m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
                  for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    Await(war[cta_m, cta_n], cuda_temporal, ~3)
                  Sm90_multicast_copy_tensor_to_smem_swizzled_2f32(A_smem[cta_m, 0:2, iter_k % 4, 0:16, 0:8, 0:32], A_tensorMap[batch, 128 * cta_m + 256 * task_m:128 + 128 * cta_m + 256 * task_m, task_k, 32 * iter_k:32 + 32 * iter_k], size0=128, size1=32, ncta=2, cta_stride=1, smem_box=(1, 64, 1, 32)) >> raw[cta_m, 0:2]
                for cta_n in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster_strided(2)):
                  Sm90_multicast_copy_tensor_to_smem_swizzled_2f32(B_smem[0:2, cta_n, iter_k % 4, 0:32, 0:8, 0:32], B_tensorMap[batch, 256 * cta_n + 512 * task_n:256 + 256 * cta_n + 512 * task_n, task_k, 32 * iter_k:32 + 32 * iter_k], size0=256, size1=32, ncta=2, cta_stride=2, smem_box=(1, 128, 1, 32)) >> raw[0:2, cta_n]
                  for cta_m in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    Arrive(cuda_temporal, 1) >> raw[cta_m, 0:2] >> raw[0:2, cta_n]
              with CudaWarps(name='consumer'):
                for cta_m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
                  for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    Await(raw[cta_m, cta_n], cuda_generic_and_async_proxy, ~0)
                    for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                      Fence(wgmma_fence_1, wgmma_fence_2)  # Fence_929
                      for mma_k in seq(0, 4):
                        Sm90_mma_async_tf32(D_rmem[cta_m, cta_n, wg_m, 0:64, 0:256], A_smem[cta_m, cta_n, iter_k % 4, 8 * wg_m:8 + 8 * wg_m, 0:8, 8 * mma_k:8 + 8 * mma_k], B_smem[cta_m, cta_n, iter_k % 4, 0:32, 0:8, 8 * mma_k:8 + 8 * mma_k], M=64, N=256)
                      Arrive(wgmma_async, 1) >> cg[cta_m, cta_n, wg_m]
                      if iter_k >= 1:
                        Await(cg[cta_m, cta_n, wg_m], cuda_in_order, 1)
                    Arrive(cuda_in_order, 1) >> war[cta_m, 0:2] >> war[0:2, cta_n]
            for iter_k in seq(1, (31 + cluster_K) / 32):
              with CudaWarps(name='producer'):
                for cta_m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
                  for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    Await(war[cta_m, cta_n], cuda_temporal, ~3)
                  Sm90_multicast_copy_tensor_to_smem_swizzled_2f32(A_smem[cta_m, 0:2, iter_k % 4, 0:16, 0:8, 0:32], A_tensorMap[batch, 128 * cta_m + 256 * task_m:128 + 128 * cta_m + 256 * task_m, task_k, 32 * iter_k:32 + 32 * iter_k], size0=128, size1=32, ncta=2, cta_stride=1, smem_box=(1, 64, 1, 32)) >> raw[cta_m, 0:2]
                for cta_n in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster_strided(2)):
                  Sm90_multicast_copy_tensor_to_smem_swizzled_2f32(B_smem[0:2, cta_n, iter_k % 4, 0:32, 0:8, 0:32], B_tensorMap[batch, 256 * cta_n + 512 * task_n:256 + 256 * cta_n + 512 * task_n, task_k, 32 * iter_k:32 + 32 * iter_k], size0=256, size1=32, ncta=2, cta_stride=2, smem_box=(1, 128, 1, 32)) >> raw[0:2, cta_n]
                  for cta_m in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    Arrive(cuda_temporal, 1) >> raw[cta_m, 0:2] >> raw[0:2, cta_n]
              with CudaWarps(name='consumer'):
                for cta_m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
                  for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    Await(raw[cta_m, cta_n], cuda_generic_and_async_proxy, ~0)
                    for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                      Fence(wgmma_fence_1, wgmma_fence_2)  # Fence_1530
                      for mma_k in seq(0, 4):
                        Sm90_mma_async_tf32(D_rmem[cta_m, cta_n, wg_m, 0:64, 0:256], A_smem[cta_m, cta_n, iter_k % 4, 8 * wg_m:8 + 8 * wg_m, 0:8, 8 * mma_k:8 + 8 * mma_k], B_smem[cta_m, cta_n, iter_k % 4, 0:32, 0:8, 8 * mma_k:8 + 8 * mma_k], M=64, N=256)
                      Arrive(wgmma_async, 1) >> cg[cta_m, cta_n, wg_m]
                      if iter_k >= 1:
                        Await(cg[cta_m, cta_n, wg_m], cuda_in_order, 1)
                    Arrive(cuda_in_order, 1) >> war[cta_m, 0:2] >> war[0:2, cta_n]
            for cta_m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
              for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                with CudaWarps(name='consumer'):
                  for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                    Await(cg[cta_m, cta_n, wg_m], cuda_in_order, 0)
            cluster_sync : barrier @CudaClusterSync
            Arrive(cuda_in_order, 1) >> cluster_sync
            for cta_m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
              for cta_n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                with CudaWarps(name='consumer'):
                  for wg_m in cuda_threads(0, 2, unit=cuda_warpgroup):
                    Sm90_mma_store_d_col_major_tf32(-(256 * task_m) - 128 * cta_m - 64 * wg_m + M, -(512 * task_n) - 256 * cta_n + N, C[batch, 256 * cta_n + 512 * task_n:256 + 256 * cta_n + 512 * task_n, 64 * wg_m + 128 * cta_m + 256 * task_m:64 + 64 * wg_m + 128 * cta_m + 256 * task_m], D_rmem[cta_m, cta_n, wg_m, 0:64, 0:256], M=64, N=256)
            Await(cluster_sync, cuda_in_order, 0)