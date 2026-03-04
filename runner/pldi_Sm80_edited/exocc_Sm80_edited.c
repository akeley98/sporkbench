#include "exocc_Sm80_edited.h"

/* Required by Sm90_SmemSwizzled(64,) */
#ifndef EXO_MEMORY_GLOBAL_Sm90_SmemSwizzled_64
#define EXO_MEMORY_GLOBAL_Sm90_SmemSwizzled_64

#ifdef __CUDACC__
template <typename T>
struct exo_Sm90_SW64 {
    T data;

    static EXO_CUDA_INLINE exo_Sm90_SW64<T>* swizzle_pointer(uintptr_t addr)
    {
        // Adapted from ThunderKittens appendix which actually documents CUDA correctly.
        uint32_t shr = uint32_t(addr) >> 3;
        const uint32_t mask = 48;
        addr = addr ^ (shr & mask);
        return reinterpret_cast<exo_Sm90_SW64*>(addr);
    }

    static __host__ __device__ constexpr uint64_t get_swizzle_bits()
    {
        return 2;
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
// CUDA device function args -- duplicated in .cuh file
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


/* relying on the following instruction..."
Sm80_cp_async_1d(dst,src,dst=f16, src=f16, size0=8)

*/

/* relying on the following instruction..."
Sm80_ldmatrix_16b(dst,src,dst=f16, src=f16, nmat0=2, nmat1=2, operand='A')

*/

/* relying on the following instruction..."
Sm80_ldmatrix_16b(dst,src,dst=f16, src=f16, nmat0=2, nmat1=2, operand='B')

*/

/* relying on the following instruction..."
Sm80_mma_m16n8(D,A,B,D=f32, A=f16, B=f16, K_pack=2)

*/

/* relying on the following instruction..."
Sm80_mma_m16n8_zero(D,D=f32)

*/

/* relying on the following instruction..."
Sm80_mma_store_d_row_major_tf32(dst,rmem)

*/
// starter_ring_smem_gemm_2(
//     L : size,
//     M : size,
//     N : size,
//     K : size,
//     C : f32[L, M, N] @CudaGmemLinear,
//     A : f16[L, M, K] @CudaGmemLinear,
//     B : f16[L, N, K] @CudaGmemLinear
// )
void starter_ring_smem_gemm_2( void *ctxt, int_fast32_t L, int_fast32_t M, int_fast32_t N, int_fast32_t K, float* C, const exo_f16* A, const exo_f16* B ) {
  EXO_ASSUME(M % 128 == 0);
  EXO_ASSUME(N % 128 == 0);
  EXO_ASSUME(K % 128 == 0);
  EXO_ASSUME(K >= 128);
  {
    struct exo_CudaDeviceArgs0_starter_ring_smem_gemm_2 exo_deviceArgs = {
      L, M, N, K, C, A, B, exo_excut_get_device_log()
    };
    exo_cudaLaunch0_starter_ring_smem_gemm_2(exo_cudaStream, &exo_deviceArgs);
  }
}


