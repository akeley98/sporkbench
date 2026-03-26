#include "exocc_Sm90a_edited_tk_attn_fwd.cuh"
__launch_bounds__(512, 1)
__global__ void
exo_deviceFunction0_edited_exo_tk_attn_fwd_Hdim128(__grid_constant__ const struct exo_CudaDeviceArgs0_edited_exo_tk_attn_fwd_Hdim128 exo_deviceArgs)
{
  extern __shared__ char exo_smem[];
  exo_ExcutThreadLog exo_excutLog = exo_excut_begin_thread_log(exo_deviceArgs.exo_excutDeviceLog);
  exo_Cuda0_edited_exo_tk_attn_fwd_Hdim128::exo_deviceSetup(exo_smem, exo_deviceArgs, exo_excutLog);
  exo_Cuda0_edited_exo_tk_attn_fwd_Hdim128::exo_deviceMainLoop(exo_smem, exo_deviceArgs, exo_excutLog);
}

void
exo_cudaLaunch0_edited_exo_tk_attn_fwd_Hdim128(cudaStream_t exo_cudaStream, const struct exo_CudaDeviceArgs0_edited_exo_tk_attn_fwd_Hdim128* exo_deviceArgs)
{
  exo_Cuda0_edited_exo_tk_attn_fwd_Hdim128::exo_cudaLaunch(exo_cudaStream, *exo_deviceArgs);
}
