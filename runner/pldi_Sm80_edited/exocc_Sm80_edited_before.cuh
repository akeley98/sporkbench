#pragma once
#include "exocc_Sm80_edited.h"
#if EXO_EXCUT_bENABLE_LOG
#include "exocc_Sm80_edited.excut_str_table"
#endif
/* Required by Sm90_SmemSwizzled(128,) */
#ifndef EXO_MEMORY_GLOBAL_Sm90_SmemSwizzled_128
#define EXO_MEMORY_GLOBAL_Sm90_SmemSwizzled_128

#ifdef __CUDACC__
template <typename T>
struct exo_Sm90_SW128 {
    T data;

    static EXO_CUDA_INLINE exo_Sm90_SW128<T>* swizzle_pointer(uintptr_t addr)
    {
        // Adapted from ThunderKittens appendix which actually documents CUDA correctly.
        uint32_t shr = uint32_t(addr) >> 3;
        const uint32_t mask = 112;
        addr = addr ^ (shr & mask);
        return reinterpret_cast<exo_Sm90_SW128*>(addr);
    }

    static __host__ __device__ constexpr uint64_t get_swizzle_bits()
    {
        return 1;
    }

    EXO_CUDA_INLINE const T& swizzle_get() const
    {
        return swizzle_pointer(reinterpret_cast<uintptr_t>(&data))->data;
    }

    EXO_CUDA_INLINE T& swizzle_get()
    {
        return swizzle_pointer(reinterpret_cast<uintptr_t>(&data))->data;
    }
};
#endif

#endif
/* Required by CudaRmemPacked32 */
#ifndef EXO_MEMORY_GLOBAL_exo_CudaRmemPacked32
#define EXO_MEMORY_GLOBAL_exo_CudaRmemPacked32

#ifdef __CUDACC__

template <typename PtxType, typename Scalar, typename PackedStruct>
struct exo_CudaRmemPacked32
{
    static_assert(sizeof(PtxType) == sizeof(PackedStruct));
    PtxType ptx_data;

    template <typename Index>
    __device__ auto operator[] (Index i) const -> Scalar
    {
        if constexpr (sizeof(Scalar) == 4)
            return ptx_data;
        else if (i == 0)
            return reinterpret_cast<const PackedStruct*>(&ptx_data)->x;
        else
            return reinterpret_cast<const PackedStruct*>(&ptx_data)->y;
    }
};

using exo_CudaRmemPacked32_f32 = exo_CudaRmemPacked32<float, float, float>;
using exo_CudaRmemPacked32_i32 = exo_CudaRmemPacked32<int32_t, int32_t, int32_t>;
using exo_CudaRmemPacked32_f16 = exo_CudaRmemPacked32<int32_t, __half, __half2>;
using exo_CudaRmemPacked32_bf16 = exo_CudaRmemPacked32<int32_t, __nv_bfloat16, __nv_bfloat162>;

#endif

#endif
/* Required by CudaSmemAtomicity16B */
#ifndef EXO_WIN_1F16
#define EXO_WIN_1F16
struct exo_win_1f16 {
    exo_f16 * const data;
    const int_fast32_t strides[1];
};
#endif
/* Required by CudaGmemAtomicity16B */
#ifndef EXO_WIN_1F16C
#define EXO_WIN_1F16C
struct exo_win_1f16c {
    const exo_f16 * const data;
    const int_fast32_t strides[1];
};
#endif

namespace exo_CudaUtil_exocc_Sm80_edited {
namespace exo_CudaUtil = ::exo_CudaUtil_exocc_Sm80_edited;
}  // end namespace exo_CudaUtil_exocc_Sm80_edited
// CUDA device function args -- duplicated in .c file
struct exo_CudaDeviceArgs0_starter_ring_smem_gemm_2
{
    int_fast32_t L;  // L : size
    int_fast32_t M;  // M : size
    int_fast32_t N;  // N : size
    int_fast32_t K;  // K : size
    float* C;  // C : f32[L, M, N] @CudaGmemLinear
    const exo_f16* A;  // A : f16[L, M, K] @CudaGmemLinear
    const exo_f16* B;  // B : f16[L, N, K] @CudaGmemLinear
    EXO_EXCUT_DEVICE_LOG_MEMBER  // for Exo pytest (exo_excut.h)
};

// We need this inline namespace to avoid ODR problems in pytest.
inline namespace exo_CudaInline_exocc_Sm80_edited {
struct exo_Cuda0_starter_ring_smem_gemm_2
{
  using exo_DeviceArgs = exo_CudaDeviceArgs0_starter_ring_smem_gemm_2;

  static constexpr uint32_t exo_blockDim = 128;
  static constexpr uint32_t exo_clusterDim = 1;

  static constexpr unsigned exo_smemBytes = 65536;
  static constexpr unsigned exo_smemOffset0_A_smem = 0;  // 32768-byte allocation
  static constexpr unsigned exo_smemOffset1_B_smem = 32768;  // 32768-byte allocation

  struct exo_Task
  {
    int_fast32_t batch;
    int_fast32_t m_task;
    int_fast32_t n_task;
  };

  struct exo_TaskGenerator
  {
    uint32_t exo_taskIndex;
    uint32_t exo_numClusters;
    uint32_t exo_taskCount;
    int_fast32_t exo_cudaTasksLo_batch;
    uint32_t exo_cudaTasksNum_batch;
    int_fast32_t exo_cudaTasksLo_m_task;
    uint32_t exo_cudaTasksNum_m_task;
    int_fast32_t exo_cudaTasksLo_n_task;
    uint32_t exo_cudaTasksNum_n_task;
    EXO_CUDA_INLINE exo_TaskGenerator(
        uint32_t cluster_index, uint32_t num_clusters,
        int_fast32_t _exo_cudaTasksLo_batch, int_fast32_t _exo_cudaTasksHi_batch,
        int_fast32_t _exo_cudaTasksLo_m_task, int_fast32_t _exo_cudaTasksHi_m_task,
        int_fast32_t _exo_cudaTasksLo_n_task, int_fast32_t _exo_cudaTasksHi_n_task,
        const exo_DeviceArgs&)
    {
      exo_taskIndex = cluster_index;
      exo_numClusters = num_clusters;
      exo_taskCount = 1;
      exo_cudaTasksLo_batch = _exo_cudaTasksLo_batch;
      exo_cudaTasksNum_batch = static_cast<uint32_t>(_exo_cudaTasksHi_batch - _exo_cudaTasksLo_batch);
      exo_taskCount *= exo_cudaTasksNum_batch;
      exo_cudaTasksLo_m_task = _exo_cudaTasksLo_m_task;
      exo_cudaTasksNum_m_task = static_cast<uint32_t>(_exo_cudaTasksHi_m_task - _exo_cudaTasksLo_m_task);
      exo_taskCount *= exo_cudaTasksNum_m_task;
      exo_cudaTasksLo_n_task = _exo_cudaTasksLo_n_task;
      exo_cudaTasksNum_n_task = static_cast<uint32_t>(_exo_cudaTasksHi_n_task - _exo_cudaTasksLo_n_task);
      exo_taskCount *= exo_cudaTasksNum_n_task;
    }
    [[nodiscard]] EXO_CUDA_INLINE bool prepare_next_task()
    {
      return exo_taskIndex < exo_taskCount;
    }
    EXO_CUDA_INLINE exo_Task get_next_task()
    {
      exo_Task exo_task;
      uint32_t exo_tmp = exo_taskIndex;
      exo_taskIndex += exo_numClusters;
      exo_task.n_task = exo_cudaTasksLo_n_task + static_cast<int_fast32_t>(exo_tmp % exo_cudaTasksNum_n_task);
      exo_tmp /= exo_cudaTasksNum_n_task;
      exo_task.m_task = exo_cudaTasksLo_m_task + static_cast<int_fast32_t>(exo_tmp % exo_cudaTasksNum_m_task);
      exo_tmp /= exo_cudaTasksNum_m_task;
      exo_task.batch = exo_cudaTasksLo_batch + static_cast<int_fast32_t>(exo_tmp % exo_cudaTasksNum_batch);
      exo_tmp /= exo_cudaTasksNum_batch;
      return exo_task;
    }
  };

  struct exo_SyncState
  {

  };

  static inline const char*& exo_FILE()
  {
    static const char* name = __FILE__;
    return name;
  }

  static void
  exo_cudaLaunch(cudaStream_t exo_cudaStream, const exo_DeviceArgs& exo_deviceArgs);

  static __device__ __forceinline__ void
  exo_deviceSetup(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs, exo_ExcutThreadLog exo_excutLog={});

  static __device__ __forceinline__ void
  exo_deviceMainLoop(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs, exo_ExcutThreadLog exo_excutLog={});

  static __device__ __forceinline__ void
  exo_deviceTask(
      char* exo_smem,
      exo_SyncState& exo_syncState,
      const exo_DeviceArgs& exo_deviceArgs,
      exo_Task exo_task,
      exo_ExcutThreadLog exo_excutLog={});
};
}  // end inline namespace

inline void
exo_CudaInline_exocc_Sm80_edited::exo_Cuda0_starter_ring_smem_gemm_2::exo_cudaLaunch(
    cudaStream_t exo_cudaStream,
    const exo_DeviceArgs& exo_deviceArgs)
{
  namespace exo_CudaUtil = exo_CudaUtil_exocc_Sm80_edited;
  cudaFuncSetAttribute(exo_deviceFunction0_starter_ring_smem_gemm_2, cudaFuncAttributeMaxDynamicSharedMemorySize, exo_smemBytes);
  // TODO how expensive is it to query this every time?
  int exo_cudaDevice;
  cudaGetDevice(&exo_cudaDevice);
  int exo_SMs;
  cudaDeviceGetAttribute(&exo_SMs, cudaDevAttrMultiProcessorCount, exo_cudaDevice);
  const unsigned exo_gridDim = (unsigned(exo_SMs) & ~(exo_clusterDim - 1)) * 2u;

  cudaLaunchConfig_t exo_launchConfig = {};
  exo_launchConfig.gridDim = dim3(exo_gridDim, 1, 1);
  exo_launchConfig.blockDim = dim3(exo_blockDim, 1, 1);
  exo_launchConfig.dynamicSmemBytes = exo_smemBytes;
  exo_launchConfig.stream = exo_cudaStream;

  cudaLaunchKernelEx(&exo_launchConfig, exo_deviceFunction0_starter_ring_smem_gemm_2, exo_deviceArgs);

  exo_excut_flush_device_log(
      exo_cudaStream, exo_gridDim, exo_blockDim,
      exo_CudaUtil::exo_excut_str_id_count, exo_CudaUtil::exo_excut_str_table,
      1, &exo_FILE());
}

__device__ __forceinline__ void
exo_CudaInline_exocc_Sm80_edited::exo_Cuda0_starter_ring_smem_gemm_2::exo_deviceSetup(
    char* exo_smem,
    const exo_DeviceArgs& exo_deviceArgs,
    exo_ExcutThreadLog exo_excutLog)
{
  // No mbarriers used
}

__device__ __forceinline__ void
exo_CudaInline_exocc_Sm80_edited::exo_Cuda0_starter_ring_smem_gemm_2::exo_deviceTask(
    char* exo_smem,
    exo_SyncState& exo_syncState,
    const exo_DeviceArgs& exo_deviceArgs,
    exo_Task exo_task,
    exo_ExcutThreadLog exo_excutLog)
{
  namespace exo_CudaUtil = exo_CudaUtil_exocc_Sm80_edited;
  float D_rmem[4][8][4];
  // cuda_threads(0, 2, unit=2 * cuda_warp)
  if ([[maybe_unused]] int exo_64thr_mw = (threadIdx.x / 64); 1) {
    // cuda_threads(0, 2, unit=cuda_warp)
    if ([[maybe_unused]] int exo_32thr_nw = (threadIdx.x % 64 / 32); 1) {
      for (int ms = 0; ms < 4; ms++) {
        for (int ns = 0; ns < 8; ns++) {
          D_rmem[ms][ns][0] = 0;
          D_rmem[ms][ns][1] = 0;
          D_rmem[ms][ns][2] = 0;
          D_rmem[ms][ns][3] = 0;
        }
      }
    }
  }
  auto& A_smem = reinterpret_cast<exo_Sm90_SW128<exo_f16> (&)[]>(exo_smem[exo_smemOffset0_A_smem]);
  auto& B_smem = reinterpret_cast<exo_Sm90_SW128<exo_f16> (&)[]>(exo_smem[exo_smemOffset1_B_smem]);
  exo_CudaRmemPacked32_f16 A_rmem[2][8][2];
  exo_CudaRmemPacked32_f16 B_rmem[2][8][2];
  // cg: barrier[128] @ CudaCommitGroup
  #pragma unroll
  for (int ks = 0; ks < 3; ks++) {
    struct exo_win_2f16c A_tile = (struct exo_win_2f16c) { &exo_deviceArgs.A[exo_task.batch * (exo_deviceArgs.M * exo_deviceArgs.K) + (128 * exo_task.m_task) * exo_deviceArgs.K + (32 * ks)], {exo_deviceArgs.K, 1} };
    struct exo_win_2f16c B_tile = (struct exo_win_2f16c) { &exo_deviceArgs.B[exo_task.batch * (exo_deviceArgs.N * exo_deviceArgs.K) + (128 * exo_task.n_task) * exo_deviceArgs.K + (32 * ks)], {exo_deviceArgs.K, 1} };
    #pragma unroll
    for (int cp_async_mno = 0; cp_async_mno < 4; cp_async_mno++) {
      // cuda_threads(0, 32, unit=4 * cuda_thread)
      if ([[maybe_unused]] int exo_4thr_cp_async_mni = (threadIdx.x / 4); 1) {
        // cuda_threads(0, 4, unit=cuda_thread)
        if ([[maybe_unused]] int exo_1thr_cp_async_ko = (threadIdx.x % 4); 1) {
          asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;"
              :
              :"r"(exo_smemU32((&A_smem[(ks * 4096 + (exo_4thr_cp_async_mni + 32 * cp_async_mno) * 32 + 8 * exo_1thr_cp_async_ko)].swizzle_get()))),
              "l"((&A_tile.data[(exo_4thr_cp_async_mni + 32 * cp_async_mno) * A_tile.strides[0] + (8 * exo_1thr_cp_async_ko) * A_tile.strides[1]]))
          );
          exo_excutLog.log_action(EXO_EXCUT_STR_ID(cp_async_cg_shared_global), 0, __LINE__);
          exo_excutLog.log_u32_arg(exo_smemU32((&A_smem[(ks * 4096 + (exo_4thr_cp_async_mni + 32 * cp_async_mno) * 32 + 8 * exo_1thr_cp_async_ko)].swizzle_get())));
          exo_excutLog.log_ptr_arg((&A_tile.data[(exo_4thr_cp_async_mni + 32 * cp_async_mno) * A_tile.strides[0] + (8 * exo_1thr_cp_async_ko) * A_tile.strides[1]]));
          exo_excutLog.log_u32_arg(static_cast<uint32_t>(16));
        }
      }
    }
    #pragma unroll
    for (int cp_async_mno = 0; cp_async_mno < 4; cp_async_mno++) {
      // cuda_threads(0, 32, unit=4 * cuda_thread)
      if ([[maybe_unused]] int exo_4thr_cp_async_mni = (threadIdx.x / 4); 1) {
        // cuda_threads(0, 4, unit=cuda_thread)
        if ([[maybe_unused]] int exo_1thr_cp_async_ko = (threadIdx.x % 4); 1) {
          asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;"
              :
              :"r"(exo_smemU32((&B_smem[(ks * 4096 + (exo_4thr_cp_async_mni + 32 * cp_async_mno) * 32 + 8 * exo_1thr_cp_async_ko)].swizzle_get()))),
              "l"((&B_tile.data[(exo_4thr_cp_async_mni + 32 * cp_async_mno) * B_tile.strides[0] + (8 * exo_1thr_cp_async_ko) * B_tile.strides[1]]))
          );
          exo_excutLog.log_action(EXO_EXCUT_STR_ID(cp_async_cg_shared_global), 0, __LINE__);
          exo_excutLog.log_u32_arg(exo_smemU32((&B_smem[(ks * 4096 + (exo_4thr_cp_async_mni + 32 * cp_async_mno) * 32 + 8 * exo_1thr_cp_async_ko)].swizzle_get())));
          exo_excutLog.log_ptr_arg((&B_tile.data[(exo_4thr_cp_async_mni + 32 * cp_async_mno) * B_tile.strides[0] + (8 * exo_1thr_cp_async_ko) * B_tile.strides[1]]));
          exo_excutLog.log_u32_arg(static_cast<uint32_t>(16));
        }
      }
    }
    // cuda_threads(0, 128, unit=cuda_thread)
    if ([[maybe_unused]] int exo_1thr_tid = threadIdx.x; 1) {
      // Arrive(Sm80_cp_async, 1) >> cg[tid]
      asm volatile(
        "cp.async.commit_group;"
      );
      exo_excutLog.log_action(EXO_EXCUT_STR_ID(cp_async_commit_group), 0, __LINE__);
    }
  }
  // cuda_threads(0, 128, unit=cuda_thread)
  if ([[maybe_unused]] int exo_1thr_tid = threadIdx.x; 1) {
    // Await(cg[tid], cuda_in_order, 2)
    asm volatile(
      "cp.async.wait_group 2;"
    );
    exo_excutLog.log_action(EXO_EXCUT_STR_ID(cp_async_wait_group), 0, __LINE__);
    exo_excutLog.log_u32_arg(static_cast<uint32_t>(2));
  }
  // Fence(cuda_in_order, cuda_in_order)
  asm volatile(
    "barrier.cta.sync 0;"
  );
  exo_excutLog.log_action(EXO_EXCUT_STR_ID(barrier_cta_sync), 0, __LINE__);
  exo_excutLog.log_u32_arg(static_cast<uint32_t>(0));
  // cuda_threads(0, 2, unit=2 * cuda_warp)
  if ([[maybe_unused]] int exo_64thr_mw = (threadIdx.x / 64); 1) {
    // cuda_threads(0, 2, unit=cuda_warp)
    if ([[maybe_unused]] int exo_32thr_nw = (threadIdx.x % 64 / 32); 1) {
      #pragma unroll
      for (int s = 0; s < 4; s++) {
        asm volatile(
          "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
            :"=r"(A_rmem[0][2 * s][0].ptx_data), "=r"(A_rmem[0][2 * s + 1][0].ptx_data), "=r"(A_rmem[0][2 * s][1].ptx_data), "=r"(A_rmem[0][2 * s + 1][1].ptx_data)
            :"r"(exo_smemU32((&A_smem[((16 * s + 64 * exo_64thr_mw + (threadIdx.x % 8) + 8 * (((threadIdx.x) >> 3) & 1)) * 32 + 8 * (((threadIdx.x) >> 4) & 1))].swizzle_get())))
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(ldmatrix_sync_aligned_x4_m8n8_shared_b16), 0, __LINE__);
        exo_excutLog.log_u32_arg(exo_smemU32((&A_smem[((16 * s + 64 * exo_64thr_mw + (threadIdx.x % 8) + 8 * (((threadIdx.x) >> 3) & 1)) * 32 + 8 * (((threadIdx.x) >> 4) & 1))].swizzle_get())));
        asm volatile(
          "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
            :"=r"(B_rmem[0][2 * s][0].ptx_data), "=r"(B_rmem[0][2 * s][1].ptx_data), "=r"(B_rmem[0][2 * s + 1][0].ptx_data), "=r"(B_rmem[0][2 * s + 1][1].ptx_data)
            :"r"(exo_smemU32((&B_smem[((16 * s + 64 * exo_32thr_nw + (threadIdx.x % 8) + 8 * (((threadIdx.x) >> 4) & 1)) * 32 + 8 * (((threadIdx.x) >> 3) & 1))].swizzle_get())))
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(ldmatrix_sync_aligned_x4_m8n8_shared_b16), 0, __LINE__);
        exo_excutLog.log_u32_arg(exo_smemU32((&B_smem[((16 * s + 64 * exo_32thr_nw + (threadIdx.x % 8) + 8 * (((threadIdx.x) >> 4) & 1)) * 32 + 8 * (((threadIdx.x) >> 3) & 1))].swizzle_get())));
      }
    }
  }
  for (int ks = 3; ks < ((exo_deviceArgs.K) / (32)) + 4 - 1; ks++) {
    // cuda_threads(0, 2, unit=2 * cuda_warp)
    if ([[maybe_unused]] int exo_64thr_mw = (threadIdx.x / 64); 1) {
      // cuda_threads(0, 2, unit=cuda_warp)
      if ([[maybe_unused]] int exo_32thr_nw = (threadIdx.x % 64 / 32); 1) {
        #pragma unroll
        for (int s = 0; s < 4; s++) {
          asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
              :"=r"(A_rmem[1][2 * s][0].ptx_data), "=r"(A_rmem[1][2 * s + 1][0].ptx_data), "=r"(A_rmem[1][2 * s][1].ptx_data), "=r"(A_rmem[1][2 * s + 1][1].ptx_data)
              :"r"(exo_smemU32((&A_smem[(((-3 + ks) & 3) * 4096 + (16 * s + 64 * exo_64thr_mw + (threadIdx.x % 8) + 8 * (((threadIdx.x) >> 3) & 1)) * 32 + 16 + 8 * (((threadIdx.x) >> 4) & 1))].swizzle_get())))
          );
          exo_excutLog.log_action(EXO_EXCUT_STR_ID(ldmatrix_sync_aligned_x4_m8n8_shared_b16), 0, __LINE__);
          exo_excutLog.log_u32_arg(exo_smemU32((&A_smem[(((-3 + ks) & 3) * 4096 + (16 * s + 64 * exo_64thr_mw + (threadIdx.x % 8) + 8 * (((threadIdx.x) >> 3) & 1)) * 32 + 16 + 8 * (((threadIdx.x) >> 4) & 1))].swizzle_get())));
          asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
              :"=r"(B_rmem[1][2 * s][0].ptx_data), "=r"(B_rmem[1][2 * s][1].ptx_data), "=r"(B_rmem[1][2 * s + 1][0].ptx_data), "=r"(B_rmem[1][2 * s + 1][1].ptx_data)
              :"r"(exo_smemU32((&B_smem[(((-3 + ks) & 3) * 4096 + (16 * s + 64 * exo_32thr_nw + (threadIdx.x % 8) + 8 * (((threadIdx.x) >> 4) & 1)) * 32 + 16 + 8 * (((threadIdx.x) >> 3) & 1))].swizzle_get())))
          );
          exo_excutLog.log_action(EXO_EXCUT_STR_ID(ldmatrix_sync_aligned_x4_m8n8_shared_b16), 0, __LINE__);
          exo_excutLog.log_u32_arg(exo_smemU32((&B_smem[(((-3 + ks) & 3) * 4096 + (16 * s + 64 * exo_32thr_nw + (threadIdx.x % 8) + 8 * (((threadIdx.x) >> 4) & 1)) * 32 + 16 + 8 * (((threadIdx.x) >> 3) & 1))].swizzle_get())));
        }
        #pragma unroll
        for (int ms = 0; ms < 4; ms++) {
          #pragma unroll
          for (int ns = 0; ns < 8; ns++) {
            asm(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                :"=f"(D_rmem[ms][ns][0]), "=f"(D_rmem[ms][ns][1]), "=f"(D_rmem[ms][ns][2]), "=f"(D_rmem[ms][ns][3])
                :"r"(*reinterpret_cast<const int32_t*>((&A_rmem[0][2 * ms][0].ptx_data))), "r"(*reinterpret_cast<const int32_t*>((&A_rmem[0][2 * ms + 1][0].ptx_data))), "r"(*reinterpret_cast<const int32_t*>((&A_rmem[0][2 * ms][1].ptx_data))), "r"(*reinterpret_cast<const int32_t*>((&A_rmem[0][2 * ms + 1][1].ptx_data))),
                "r"(*reinterpret_cast<const int32_t*>((&B_rmem[0][ns][0].ptx_data))), "r"(*reinterpret_cast<const int32_t*>((&B_rmem[0][ns][1].ptx_data))),
                "f"(D_rmem[ms][ns][0]), "f"(D_rmem[ms][ns][1]), "f"(D_rmem[ms][ns][2]), "f"(D_rmem[ms][ns][3])
            );
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(mma_sync_aligned_m16n8k16_row_col_f32_f16_f16_f32), 0, __LINE__);
          }
        }
      }
    }
    if (ks < ((exo_deviceArgs.K) / (32))) {
      struct exo_win_2f16c A_tile = (struct exo_win_2f16c) { &exo_deviceArgs.A[exo_task.batch * (exo_deviceArgs.M * exo_deviceArgs.K) + (128 * exo_task.m_task) * exo_deviceArgs.K + (32 * ks)], {exo_deviceArgs.K, 1} };
      struct exo_win_2f16c B_tile = (struct exo_win_2f16c) { &exo_deviceArgs.B[exo_task.batch * (exo_deviceArgs.N * exo_deviceArgs.K) + (128 * exo_task.n_task) * exo_deviceArgs.K + (32 * ks)], {exo_deviceArgs.K, 1} };
      #pragma unroll
      for (int cp_async_mno = 0; cp_async_mno < 4; cp_async_mno++) {
        // cuda_threads(0, 32, unit=4 * cuda_thread)
        if ([[maybe_unused]] int exo_4thr_cp_async_mni = (threadIdx.x / 4); 1) {
          // cuda_threads(0, 4, unit=cuda_thread)
          if ([[maybe_unused]] int exo_1thr_cp_async_ko = (threadIdx.x % 4); 1) {
            asm volatile(
              "cp.async.cg.shared.global [%0], [%1], 16;"
                :
                :"r"(exo_smemU32((&A_smem[((ks & 3) * 4096 + (exo_4thr_cp_async_mni + 32 * cp_async_mno) * 32 + 8 * exo_1thr_cp_async_ko)].swizzle_get()))),
                "l"((&A_tile.data[(exo_4thr_cp_async_mni + 32 * cp_async_mno) * A_tile.strides[0] + (8 * exo_1thr_cp_async_ko) * A_tile.strides[1]]))
            );
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(cp_async_cg_shared_global), 0, __LINE__);
            exo_excutLog.log_u32_arg(exo_smemU32((&A_smem[((ks & 3) * 4096 + (exo_4thr_cp_async_mni + 32 * cp_async_mno) * 32 + 8 * exo_1thr_cp_async_ko)].swizzle_get())));
            exo_excutLog.log_ptr_arg((&A_tile.data[(exo_4thr_cp_async_mni + 32 * cp_async_mno) * A_tile.strides[0] + (8 * exo_1thr_cp_async_ko) * A_tile.strides[1]]));
            exo_excutLog.log_u32_arg(static_cast<uint32_t>(16));
          }
        }
      }
      #pragma unroll
      for (int cp_async_mno = 0; cp_async_mno < 4; cp_async_mno++) {
        // cuda_threads(0, 32, unit=4 * cuda_thread)
        if ([[maybe_unused]] int exo_4thr_cp_async_mni = (threadIdx.x / 4); 1) {
          // cuda_threads(0, 4, unit=cuda_thread)
          if ([[maybe_unused]] int exo_1thr_cp_async_ko = (threadIdx.x % 4); 1) {
            asm volatile(
              "cp.async.cg.shared.global [%0], [%1], 16;"
                :
                :"r"(exo_smemU32((&B_smem[((ks & 3) * 4096 + (exo_4thr_cp_async_mni + 32 * cp_async_mno) * 32 + 8 * exo_1thr_cp_async_ko)].swizzle_get()))),
                "l"((&B_tile.data[(exo_4thr_cp_async_mni + 32 * cp_async_mno) * B_tile.strides[0] + (8 * exo_1thr_cp_async_ko) * B_tile.strides[1]]))
            );
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(cp_async_cg_shared_global), 0, __LINE__);
            exo_excutLog.log_u32_arg(exo_smemU32((&B_smem[((ks & 3) * 4096 + (exo_4thr_cp_async_mni + 32 * cp_async_mno) * 32 + 8 * exo_1thr_cp_async_ko)].swizzle_get())));
            exo_excutLog.log_ptr_arg((&B_tile.data[(exo_4thr_cp_async_mni + 32 * cp_async_mno) * B_tile.strides[0] + (8 * exo_1thr_cp_async_ko) * B_tile.strides[1]]));
            exo_excutLog.log_u32_arg(static_cast<uint32_t>(16));
          }
        }
      }
    }
    // cuda_threads(0, 128, unit=cuda_thread)
    if ([[maybe_unused]] int exo_1thr_tid = threadIdx.x; 1) {
      // Arrive(Sm80_cp_async, 1) >> cg[tid]
      asm volatile(
        "cp.async.commit_group;"
      );
      exo_excutLog.log_action(EXO_EXCUT_STR_ID(cp_async_commit_group), 0, __LINE__);
      // Await(cg[tid], cuda_in_order, 2)
      asm volatile(
        "cp.async.wait_group 2;"
      );
      exo_excutLog.log_action(EXO_EXCUT_STR_ID(cp_async_wait_group), 0, __LINE__);
      exo_excutLog.log_u32_arg(static_cast<uint32_t>(2));
    }
    // Fence(cuda_in_order, cuda_in_order)
    asm volatile(
      "barrier.cta.sync 0;"
    );
    exo_excutLog.log_action(EXO_EXCUT_STR_ID(barrier_cta_sync), 0, __LINE__);
    exo_excutLog.log_u32_arg(static_cast<uint32_t>(0));
    // cuda_threads(0, 2, unit=2 * cuda_warp)
    if ([[maybe_unused]] int exo_64thr_mw = (threadIdx.x / 64); 1) {
      // cuda_threads(0, 2, unit=cuda_warp)
      if ([[maybe_unused]] int exo_32thr_nw = (threadIdx.x % 64 / 32); 1) {
        #pragma unroll
        for (int s = 0; s < 4; s++) {
          asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
              :"=r"(A_rmem[0][2 * s][0].ptx_data), "=r"(A_rmem[0][2 * s + 1][0].ptx_data), "=r"(A_rmem[0][2 * s][1].ptx_data), "=r"(A_rmem[0][2 * s + 1][1].ptx_data)
              :"r"(exo_smemU32((&A_smem[(((-2 + ks) & 3) * 4096 + (16 * s + 64 * exo_64thr_mw + (threadIdx.x % 8) + 8 * (((threadIdx.x) >> 3) & 1)) * 32 + 8 * (((threadIdx.x) >> 4) & 1))].swizzle_get())))
          );
          exo_excutLog.log_action(EXO_EXCUT_STR_ID(ldmatrix_sync_aligned_x4_m8n8_shared_b16), 0, __LINE__);
          exo_excutLog.log_u32_arg(exo_smemU32((&A_smem[(((-2 + ks) & 3) * 4096 + (16 * s + 64 * exo_64thr_mw + (threadIdx.x % 8) + 8 * (((threadIdx.x) >> 3) & 1)) * 32 + 8 * (((threadIdx.x) >> 4) & 1))].swizzle_get())));
          asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
              :"=r"(B_rmem[0][2 * s][0].ptx_data), "=r"(B_rmem[0][2 * s][1].ptx_data), "=r"(B_rmem[0][2 * s + 1][0].ptx_data), "=r"(B_rmem[0][2 * s + 1][1].ptx_data)
              :"r"(exo_smemU32((&B_smem[(((-2 + ks) & 3) * 4096 + (16 * s + 64 * exo_32thr_nw + (threadIdx.x % 8) + 8 * (((threadIdx.x) >> 4) & 1)) * 32 + 8 * (((threadIdx.x) >> 3) & 1))].swizzle_get())))
          );
          exo_excutLog.log_action(EXO_EXCUT_STR_ID(ldmatrix_sync_aligned_x4_m8n8_shared_b16), 0, __LINE__);
          exo_excutLog.log_u32_arg(exo_smemU32((&B_smem[(((-2 + ks) & 3) * 4096 + (16 * s + 64 * exo_32thr_nw + (threadIdx.x % 8) + 8 * (((threadIdx.x) >> 4) & 1)) * 32 + 8 * (((threadIdx.x) >> 3) & 1))].swizzle_get())));
        }
        #pragma unroll
        for (int ms = 0; ms < 4; ms++) {
          #pragma unroll
          for (int ns = 0; ns < 8; ns++) {
            asm(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                :"=f"(D_rmem[ms][ns][0]), "=f"(D_rmem[ms][ns][1]), "=f"(D_rmem[ms][ns][2]), "=f"(D_rmem[ms][ns][3])
                :"r"(*reinterpret_cast<const int32_t*>((&A_rmem[1][2 * ms][0].ptx_data))), "r"(*reinterpret_cast<const int32_t*>((&A_rmem[1][2 * ms + 1][0].ptx_data))), "r"(*reinterpret_cast<const int32_t*>((&A_rmem[1][2 * ms][1].ptx_data))), "r"(*reinterpret_cast<const int32_t*>((&A_rmem[1][2 * ms + 1][1].ptx_data))),
                "r"(*reinterpret_cast<const int32_t*>((&B_rmem[1][ns][0].ptx_data))), "r"(*reinterpret_cast<const int32_t*>((&B_rmem[1][ns][1].ptx_data))),
                "f"(D_rmem[ms][ns][0]), "f"(D_rmem[ms][ns][1]), "f"(D_rmem[ms][ns][2]), "f"(D_rmem[ms][ns][3])
            );
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(mma_sync_aligned_m16n8k16_row_col_f32_f16_f16_f32), 0, __LINE__);
          }
        }
      }
    }
  }
  // cuda_threads(0, 128, unit=cuda_thread)
  if ([[maybe_unused]] int exo_1thr_tid = threadIdx.x; 1) {
    // Arrive(Sm80_cp_async, 1) >> cg[tid]
    asm volatile(
      "cp.async.commit_group;"
    );
    exo_excutLog.log_action(EXO_EXCUT_STR_ID(cp_async_commit_group), 0, __LINE__);
    // Await(cg[tid], cuda_in_order, 0)
    asm volatile(
      "cp.async.wait_group 0;"
    );
    exo_excutLog.log_action(EXO_EXCUT_STR_ID(cp_async_wait_group), 0, __LINE__);
    exo_excutLog.log_u32_arg(static_cast<uint32_t>(0));
  }
  // free(cg)
  // Fence(cuda_in_order, cuda_in_order)
  asm volatile(
    "barrier.cta.sync 0;"
  );
  exo_excutLog.log_action(EXO_EXCUT_STR_ID(barrier_cta_sync), 0, __LINE__);
  exo_excutLog.log_u32_arg(static_cast<uint32_t>(0));
  // cuda_threads(0, 2, unit=2 * cuda_warp)
  if ([[maybe_unused]] int exo_64thr_mw = (threadIdx.x / 64); 1) {
    // cuda_threads(0, 2, unit=cuda_warp)
    if ([[maybe_unused]] int exo_32thr_nw = (threadIdx.x % 64 / 32); 1) {
      struct exo_win_2f32 C_tile = (struct exo_win_2f32) { &exo_deviceArgs.C[exo_task.batch * (exo_deviceArgs.M * exo_deviceArgs.N) + (64 * exo_64thr_mw + 128 * exo_task.m_task) * exo_deviceArgs.N + (64 * exo_32thr_nw + 128 * exo_task.n_task)], {exo_deviceArgs.N, 1} };
      for (int ms = 0; ms < 4; ms++) {
        for (int ns = 0; ns < 8; ns++) {
          {
            const unsigned exo_lane = threadIdx.x % 32;
            const unsigned exo_m = exo_lane / 4;
            const unsigned exo_n = (exo_lane % 4) * 2;
            C_tile.data[(16 * ms + (exo_m + 0)) * C_tile.strides[0] + (8 * ns + (exo_n + 0)) * C_tile.strides[1]] = D_rmem[ms][ns][0];
            C_tile.data[(16 * ms + (exo_m + 0)) * C_tile.strides[0] + (8 * ns + (exo_n + 1)) * C_tile.strides[1]] = D_rmem[ms][ns][1];
            C_tile.data[(16 * ms + (exo_m + 8)) * C_tile.strides[0] + (8 * ns + (exo_n + 0)) * C_tile.strides[1]] = D_rmem[ms][ns][2];
            C_tile.data[(16 * ms + (exo_m + 8)) * C_tile.strides[0] + (8 * ns + (exo_n + 1)) * C_tile.strides[1]] = D_rmem[ms][ns][3];
          }
        }
      }
    }
  }
}
__device__ __forceinline__ void
exo_CudaInline_exocc_Sm80_edited::exo_Cuda0_starter_ring_smem_gemm_2::exo_deviceMainLoop(
    char* exo_smem,
    const exo_DeviceArgs& exo_deviceArgs,
    exo_ExcutThreadLog exo_excutLog)
{
  namespace exo_CudaUtil = exo_CudaUtil_exocc_Sm80_edited;
  exo_SyncState exo_syncState{};
  int32_t nreg;
  nreg = ((int32_t) 0);
  if (nreg == ((int32_t) 0)) {
    if (int tmp = threadIdx.x; tmp >= 0 && tmp < 128) {
      {
        int_fast32_t exo_cudaTasksLo_batch = 0;
        int_fast32_t exo_cudaTasksHi_batch = exo_deviceArgs.L;
        {
          int_fast32_t exo_cudaTasksLo_m_task = 0;
          int_fast32_t exo_cudaTasksHi_m_task = ((127 + exo_deviceArgs.M) / (128));
          {
            int_fast32_t exo_cudaTasksLo_n_task = 0;
            int_fast32_t exo_cudaTasksHi_n_task = ((127 + exo_deviceArgs.N) / (128));
            exo_TaskGenerator exo_taskGenerator(
                blockIdx.x / exo_clusterDim,
                gridDim.x / exo_clusterDim,
                exo_cudaTasksLo_batch, exo_cudaTasksHi_batch,
                exo_cudaTasksLo_m_task, exo_cudaTasksHi_m_task,
                exo_cudaTasksLo_n_task, exo_cudaTasksHi_n_task,
                exo_deviceArgs);
            while (exo_taskGenerator.prepare_next_task()) {
              exo_deviceTask(exo_smem, exo_syncState, exo_deviceArgs, exo_taskGenerator.get_next_task(), exo_excutLog);
            }
          }
        }
      }
    }
  }
}
