#include "exocc_Sm80_edited.cuh"
__launch_bounds__(128, 2)
__global__ void
exo_deviceFunction0_starter_ring_smem_gemm_2(__grid_constant__ const struct exo_CudaDeviceArgs0_starter_ring_smem_gemm_2 exo_deviceArgs)
{
  extern __shared__ char exo_smem[];
  exo_ExcutThreadLog exo_excutLog = exo_excut_begin_thread_log(exo_deviceArgs.exo_excutDeviceLog);
  exo_Cuda0_starter_ring_smem_gemm_2::exo_deviceSetup(exo_smem, exo_deviceArgs, exo_excutLog);
  exo_Cuda0_starter_ring_smem_gemm_2::exo_deviceMainLoop(exo_smem, exo_deviceArgs, exo_excutLog);
}

void
exo_cudaLaunch0_starter_ring_smem_gemm_2(cudaStream_t exo_cudaStream, const struct exo_CudaDeviceArgs0_starter_ring_smem_gemm_2* exo_deviceArgs)
{
  exo_Cuda0_starter_ring_smem_gemm_2::exo_cudaLaunch(exo_cudaStream, *exo_deviceArgs);
}
