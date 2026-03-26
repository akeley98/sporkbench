#pragma once
#ifndef EXOCC_SM90A_EDITED_TK_ATTN_FWD_CAUSAL_H
#define EXOCC_SM90A_EDITED_TK_ATTN_FWD_CAUSAL_H


#include <stdint.h>
#include <stdbool.h>

#ifndef EXO_CUDA_HEADER_COMMON
#define EXO_CUDA_HEADER_COMMON
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifdef __CUDACC__
#define EXO_CUDA_INLINE __device__ __forceinline__
EXO_CUDA_INLINE unsigned exo_smemU32(const void* smem_ptr)
{
    return (unsigned)__cvta_generic_to_shared(smem_ptr);
}
EXO_CUDA_INLINE unsigned exo_mapa_shared_cluster(unsigned addr_u32, unsigned cta_rank)
{
#if __CUDA_ARCH__ >= 900
    asm("mapa.shared::cluster.u32 %0, %1, %2;": "=r"(addr_u32) : "r"(addr_u32), "r"(cta_rank));
#endif
    return addr_u32;
}
#endif  // __CUDACC__

#ifndef EXO_EXCUT_bENABLE_LOG
#define EXO_EXCUT_bENABLE_LOG 0
#endif

#if EXO_EXCUT_bENABLE_LOG
#include "exo_excut.h"  // Used for exo excut tests (tracing)
#else
// Do-nothing replacements for exo_excut.h
#define exo_excut_log_file_enabled() 0
#define exo_excut_begin_log_action(action_name)
#define exo_excut_log_str_arg(str)
#define exo_excut_log_int_arg(bytes, binary)
#define exo_excut_log_ptr_arg(ptr)
#define exo_excut_end_log_action(device_name, _blockIdx, _threadIdx, file, line)
#define exo_excut_get_device_log()
#define exo_excut_flush_device_log(stream, _gridDim, _blockDim, string_id_count, string_table, file_id_count, file_table)
#define EXO_EXCUT_DEVICE_LOG_MEMBER
#define EXO_EXCUT_STR_ID(c) 0
#ifdef __CUDACC__
struct exo_ExcutThreadLog {
    EXO_CUDA_INLINE void log_action(uint32_t, uint32_t, uint32_t) {}
    EXO_CUDA_INLINE void log_str_id_arg(uint32_t) {}
    EXO_CUDA_INLINE void log_u32_arg(uint32_t) {}
    EXO_CUDA_INLINE void log_u64_arg(uint32_t) {}
    EXO_CUDA_INLINE void log_ptr_arg(const void*) {}
    template <typename T>
    EXO_CUDA_INLINE void log_ptr_data_arg(const T*, uint32_t = 0) {}
};
#define exo_excut_begin_thread_log(log) {}
#endif
#endif // EXO_EXCUT_bENABLE_LOG

#endif // EXO_CUDA_HEADER_COMMON

#ifndef EXO_CUDA_STREAM_GUARD
#define EXO_CUDA_STREAM_GUARD
static const cudaStream_t exo_cudaStream = 0;
#endif
// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif



#ifdef __cplusplus
extern "C" {
#endif


/* Required by DRAM */
#ifndef EXO_MEMORY_GLOBAL_DRAM
#define EXO_MEMORY_GLOBAL_DRAM
#include <stdio.h>
#include <stdlib.h>

#endif
/* Required by CudaGmemLinear */
#ifndef EXO_MEMORY_GLOBAL_CudaGmemLinear
#define EXO_MEMORY_GLOBAL_CudaGmemLinear

#ifndef exo_cudaMallocAsync
#ifndef __cplusplus
static
#endif
inline void* exo_cudaMallocAsync_default(size_t size, cudaStream_t exo_cudaStream,
                                         const char* file __attribute__((unused)),
                                         int line __attribute__((unused)) )
{
    void* out;
    cudaMallocAsync(&out, size, exo_cudaStream);
    if (exo_excut_log_file_enabled()) {
        exo_excut_begin_log_action("cudaMallocAsync");
        exo_excut_log_ptr_arg(out);
        exo_excut_log_ptr_arg((void*)(size));
        exo_excut_log_ptr_arg(exo_cudaStream);
        exo_excut_end_log_action("cpu", 0, 0, file, line);
    }
    return out;
}
#define exo_cudaMallocAsync(size, stream) exo_cudaMallocAsync_default(size, stream, __FILE__, __LINE__)
#endif

#ifndef exo_cudaFreeAsync
#ifndef __cplusplus
static
#endif
inline void exo_cudaFreeAsync_default(void* ptr, cudaStream_t exo_cudaStream,
                                      const char* file __attribute__((unused)),
                                      int line __attribute__((unused)) )
{
    cudaFreeAsync(ptr, exo_cudaStream);
    if (exo_excut_log_file_enabled()) {
        exo_excut_begin_log_action("cudaFreeAsync");
        exo_excut_log_ptr_arg(ptr);
        exo_excut_log_ptr_arg(exo_cudaStream);
        exo_excut_end_log_action("cpu", 0, 0, file, line);
    }
}
#define exo_cudaFreeAsync(ptr, stream) exo_cudaFreeAsync_default(ptr, stream, __FILE__, __LINE__)
#endif

#endif
/* Required by CudaGmemLinear */
/* Required by Sm90_SmemSwizzled(128,) */
/* Required by Sm90_TkRmemTileA(128,) */
/* Required by Sm90_tensorMap(128, 1, 1, 1, 64, 64) */
/* Required by Sm90_tensorMap(128, 1, 1, 128, 64) */
#ifndef EXO_MEMORY_GLOBAL_exo_bf16
#define EXO_MEMORY_GLOBAL_exo_bf16
#ifndef exo_bf16  /* Define before inclusion to override exo_bf16 */
#ifdef __CUDACC__
using exo_bf16 = __nv_bfloat16;
#else
typedef struct { short bits; } exo_bf16;
#endif
#endif

#endif
/* Required by CudaGmemLinear */
#ifndef EXO_WIN_5BF16
#define EXO_WIN_5BF16
struct exo_win_5bf16 {
    exo_bf16 * const data;
    const int_fast32_t strides[5];
};
#endif
/* Required by CudaGmemLinear */
#ifndef EXO_WIN_4F32
#define EXO_WIN_4F32
struct exo_win_4f32 {
    float * const data;
    const int_fast32_t strides[4];
};
#endif
/* Required by CudaGmemLinear */
#ifndef EXO_WIN_5BF16C
#define EXO_WIN_5BF16C
struct exo_win_5bf16c {
    const exo_bf16 * const data;
    const int_fast32_t strides[5];
};
#endif
/* Required by CudaGmemLinear */
#ifndef EXO_WIN_4BF16C
#define EXO_WIN_4BF16C
struct exo_win_4bf16c {
    const exo_bf16 * const data;
    const int_fast32_t strides[4];
};
#endif
// edited_exo_tk_attn_fwd_Hdim128_causal(
//     Batch : size,
//     KV_Heads : size,
//     Groups : size,
//     SeqLen : size,
//     O : bf16[Batch, KV_Heads, Groups, SeqLen, 128] @CudaGmemLinear,
//     lse : f32[Batch, KV_Heads, Groups, SeqLen] @CudaGmemLinear,
//     Q : bf16[Batch, KV_Heads, Groups, SeqLen, 128] @CudaGmemLinear,
//     K : bf16[Batch, KV_Heads, SeqLen, 128] @CudaGmemLinear,
//     V : bf16[Batch, KV_Heads, SeqLen, 128] @CudaGmemLinear
// )
void edited_exo_tk_attn_fwd_Hdim128_causal( void *ctxt, int_fast32_t Batch, int_fast32_t KV_Heads, int_fast32_t Groups, int_fast32_t SeqLen, exo_bf16* O, float* lse, const exo_bf16* Q, const exo_bf16* K, const exo_bf16* V );



struct exo_CudaDeviceArgs0_edited_exo_tk_attn_fwd_Hdim128_causal;

#ifdef __CUDACC__
__global__ void exo_deviceFunction0_edited_exo_tk_attn_fwd_Hdim128_causal(__grid_constant__ const struct exo_CudaDeviceArgs0_edited_exo_tk_attn_fwd_Hdim128_causal exo_deviceArgs);
#endif
void exo_cudaLaunch0_edited_exo_tk_attn_fwd_Hdim128_causal(cudaStream_t exo_cudaStream, const struct exo_CudaDeviceArgs0_edited_exo_tk_attn_fwd_Hdim128_causal* exo_deviceArgs);




#ifdef __cplusplus
}
#endif


#endif  // EXOCC_SM90A_EDITED_TK_ATTN_FWD_CAUSAL_H
