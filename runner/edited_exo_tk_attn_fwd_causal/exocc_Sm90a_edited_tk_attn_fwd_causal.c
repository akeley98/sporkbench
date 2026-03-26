#include "exocc_Sm90a_edited_tk_attn_fwd_causal.h"

/* Required by Sm90_tensorMap(0, 1, 1, 1, 64) */
/* Required by Sm90_tensorMap(128, 1, 1, 1, 64, 64) */
/* Required by Sm90_tensorMap(128, 1, 1, 128, 64) */
#include "cuda.h"
/* Required by Sm90_tensorMap(0, 1, 1, 1, 64) */
/* Required by Sm90_tensorMap(128, 1, 1, 1, 64, 64) */
/* Required by Sm90_tensorMap(128, 1, 1, 128, 64) */
#include <assert.h>
/* Required by Sm90_tensorMap(0, 1, 1, 1, 64) */
/* Required by Sm90_tensorMap(128, 1, 1, 1, 64, 64) */
/* Required by Sm90_tensorMap(128, 1, 1, 128, 64) */
#include <stdio.h>
/* Required by Sm90_tensorMap(128, 1, 1, 1, 64, 64) */
#ifndef EXO_MEMORY_GLOBAL_exo_Sm90_CUtensorMap_5_strides
#define EXO_MEMORY_GLOBAL_exo_Sm90_CUtensorMap_5_strides
typedef struct exo_Sm90_CUtensorMap_5_strides {
    // Stored in reverse-order as the raw CUtensorMap,
    // and in element count, not in bytes.
    // Leftmost stride is most-significant.
    unsigned C_strides[5];
} exo_Sm90_CUtensorMap_5_strides;

#endif
/* Required by Sm90_tensorMap(128, 1, 1, 1, 64, 64) */
#ifndef EXO_MEMORY_GLOBAL_exo_Sm90_CUtensorMap_5_dim
#define EXO_MEMORY_GLOBAL_exo_Sm90_CUtensorMap_5_dim
typedef struct exo_Sm90_CUtensorMap_5_dim {
    // Stored in the reverse-order as the raw CUtensorMap.
    // Leftmost dimension is the most-significant.
    unsigned C_dim[5];
} exo_Sm90_CUtensorMap_5_dim;

#endif
/* Required by Sm90_tensorMap(128, 1, 1, 1, 64, 64) */
#ifndef EXO_MEMORY_GLOBAL_exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64
#define EXO_MEMORY_GLOBAL_exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64
typedef struct exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64 {
    // Stored in reverse-order as the raw CUtensorMap.
    // Leftmost offset is most-significant.
    unsigned C_offsets[5];
} exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64;

#endif
/* Required by Sm90_tensorMap(0, 1, 1, 1, 64) */
/* Required by Sm90_tensorMap(128, 1, 1, 128, 64) */
#ifndef EXO_MEMORY_GLOBAL_exo_Sm90_CUtensorMap_4_strides
#define EXO_MEMORY_GLOBAL_exo_Sm90_CUtensorMap_4_strides
typedef struct exo_Sm90_CUtensorMap_4_strides {
    // Stored in reverse-order as the raw CUtensorMap,
    // and in element count, not in bytes.
    // Leftmost stride is most-significant.
    unsigned C_strides[4];
} exo_Sm90_CUtensorMap_4_strides;

#endif
/* Required by Sm90_tensorMap(0, 1, 1, 1, 64) */
/* Required by Sm90_tensorMap(128, 1, 1, 128, 64) */
#ifndef EXO_MEMORY_GLOBAL_exo_Sm90_CUtensorMap_4_dim
#define EXO_MEMORY_GLOBAL_exo_Sm90_CUtensorMap_4_dim
typedef struct exo_Sm90_CUtensorMap_4_dim {
    // Stored in the reverse-order as the raw CUtensorMap.
    // Leftmost dimension is the most-significant.
    unsigned C_dim[4];
} exo_Sm90_CUtensorMap_4_dim;

#endif
/* Required by Sm90_tensorMap(128, 1, 1, 128, 64) */
#ifndef EXO_MEMORY_GLOBAL_exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64
#define EXO_MEMORY_GLOBAL_exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64
typedef struct exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64 {
    // Stored in reverse-order as the raw CUtensorMap.
    // Leftmost offset is most-significant.
    unsigned C_offsets[4];
} exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64;

#endif
/* Required by Sm90_tensorMap(0, 1, 1, 1, 64) */
#ifndef EXO_MEMORY_GLOBAL_exo_win_4f32_Sm90_tensorMap_0_1_1_1_64
#define EXO_MEMORY_GLOBAL_exo_win_4f32_Sm90_tensorMap_0_1_1_1_64
typedef struct exo_win_4f32_Sm90_tensorMap_0_1_1_1_64 {
    // Stored in reverse-order as the raw CUtensorMap.
    // Leftmost offset is most-significant.
    unsigned C_offsets[4];
} exo_win_4f32_Sm90_tensorMap_0_1_1_1_64;

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

    static __host__ __device__ constexpr int get_swizzle_bytes()
    {
        return 128;
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

// Element type for Sm90_SmemSwizzled(128) allocations where the
// last 3 array extents are [TileOuterCols, TileRows, TileInnerCols]
// (left-padded with 1 for 0D, 1D, 2D allocations).
template <typename T, int TileOuterCols, int TileRows, int TileInnerCols>
struct exo_Sm90_SW128_tiled: public exo_Sm90_SW128<T>
{
    // Reinterpret-cast to kittens::st_subtile view of this shared memory.
    //
    // For better or worse, kittens 2D tile is expressed as a 3D tile in Exo-GPU.
    // `this` is assumed to point-to the NON-SWIZZLED base address of a complete
    // 3D tile. This returns a subtile view whose
    //
    // * size/extents are given by the Subtile template parameters
    //
    // * base offset from the `this` tile is given by runtime int offsets
    //
    // Assume for this discussion that the tile is row-major. Then,
    //
    // * Let swizzle_elements = 128 / sizeof(T)
    //
    // * The 3D Exo-GPU tile is of size [TileOuterCols, TileRows, TileInnerCols],
    //   where TileInnerCols == swizzle_elements. If not, then this tile
    //   is incompatible with ThunderKittens, TMA, and wgmma.
    //
    // * The value at coordinates (r, c) in the 2D [Rows, Cols] tile is stored at
    //   address swizzle_pointer(
    //          tile_base_addr
    //          + sizeof(T) * (c / TileInnerCols) * TileRows * TileInnerCols
    //          + sizeof(T) * (r) * TileInnerCols
    //          + sizeof(T) * (c % TileInnerCols))
    //   Basically, the column is simultaneously the fastest (%) and slowest (/) dimension
    //
    // TMA: If TileOuterCols != 1, then ThunderKittens is capable of doing a single TMA
    // copy to load/store the tile, but Exo-GPU may have to issue multiple.
    template <
        int SubtileOuterCols,
        int SubtileRows,
        int SubtileInnerCols,
        template <int, int, bool, int> class st_typed>
    EXO_CUDA_INLINE auto as_tk_subtile(int col_outer_offset, int row_offset, int col_inner_offset) const
    {
        static_assert(
            TileInnerCols * sizeof(T) == 128,
            "Exo-GPU instr didn't assert strides properly."
            " This is needed to match kittens swizzle automation"
        );
        static_assert(
            SubtileInnerCols == TileInnerCols || SubtileOuterCols == 1,
            "Exo-GPU instr used wrong SubtileInnerCols (window size last dimension)."
            " This is needed for Exo-GPU and kittens to agree on column tiling"
        );
        using st_t = st_typed<TileRows, TileOuterCols * TileInnerCols, true, 0>;
        st_t* p_tile = const_cast<st_t*>(reinterpret_cast<const st_t*>(this));
        // Note, subtile implicitly multiplies (r, c) by the subtile size.
        // We have to work around this!
        // Also, can't mention ::kittens here, because it may not be included.
        // Also also, explicit kittens swizzle is broken in this function.
        auto subtile = p_tile->template subtile<SubtileRows, SubtileOuterCols * SubtileInnerCols>(int2(0, 0));
        subtile.row_offset = static_cast<int>(row_offset);
        subtile.col_offset = static_cast<int>(col_outer_offset) * TileInnerCols + static_cast<int>(col_inner_offset);
        return subtile;
    }
};

#endif

#endif
/* Required by Sm90_tensorMap(128, 1, 1, 1, 64, 64) */
#ifndef EXO_MEMORY_GLOBAL_exo_win_2bf16_Sm90_tensorMap_128_1_1_1_64_64
#define EXO_MEMORY_GLOBAL_exo_win_2bf16_Sm90_tensorMap_128_1_1_1_64_64
typedef struct exo_win_2bf16_Sm90_tensorMap_128_1_1_1_64_64 {
    // Stored in reverse-order as the raw CUtensorMap.
    // Leftmost offset is most-significant.
    unsigned C_offsets[5];
} exo_win_2bf16_Sm90_tensorMap_128_1_1_1_64_64;

#endif
/* Required by Sm90_tensorMap(128, 1, 1, 128, 64) */
#ifndef EXO_MEMORY_GLOBAL_exo_win_2bf16_Sm90_tensorMap_128_1_1_128_64
#define EXO_MEMORY_GLOBAL_exo_win_2bf16_Sm90_tensorMap_128_1_1_128_64
typedef struct exo_win_2bf16_Sm90_tensorMap_128_1_1_128_64 {
    // Stored in reverse-order as the raw CUtensorMap.
    // Leftmost offset is most-significant.
    unsigned C_offsets[4];
} exo_win_2bf16_Sm90_tensorMap_128_1_1_128_64;

#endif
/* Required by CudaSmemLinear */
#ifndef EXO_WIN_2F32
#define EXO_WIN_2F32
struct exo_win_2f32 {
    float * const data;
    const int_fast32_t strides[2];
};
#endif
/* Required by CudaTkWarpTile(16, 128, 'row') */
/* Required by Sm90_TkRmemTileA(128,) */
/* Required by Sm90_TkRmemTileD(128,) */
#ifndef EXO_MEMORY_GLOBAL_exo_CudaTkScaleD
#define EXO_MEMORY_GLOBAL_exo_CudaTkScaleD
#ifdef __cplusplus
template <typename Tile>
struct exo_CudaTkScaleD: public Tile
{
    // Set to 0 to trigger a zero-clear on the next async mma instr.
    // Each async mma instr resets this to 1.
    int scale_d = 1;
    Tile tile;
};
#endif

#endif
/* Required by CudaSmemLinear */
#ifndef EXO_WIN_1F32
#define EXO_WIN_1F32
struct exo_win_1f32 {
    float * const data;
    const int_fast32_t strides[1];
};
#endif
/* Required by Sm90_tensorMap(0, 1, 1, 1, 64) */
#ifndef EXO_MEMORY_GLOBAL_exo_win_1f32_Sm90_tensorMap_0_1_1_1_64
#define EXO_MEMORY_GLOBAL_exo_win_1f32_Sm90_tensorMap_0_1_1_1_64
typedef struct exo_win_1f32_Sm90_tensorMap_0_1_1_1_64 {
    // Stored in reverse-order as the raw CUtensorMap.
    // Leftmost offset is most-significant.
    unsigned C_offsets[4];
} exo_win_1f32_Sm90_tensorMap_0_1_1_1_64;

#endif
/* Required by CudaSmemLinear */
#ifndef EXO_WIN_1F32C
#define EXO_WIN_1F32C
struct exo_win_1f32c {
    const float * const data;
    const int_fast32_t strides[1];
};
#endif
/* Required by Sm90_tensorMap(128, 1, 1, 1, 64, 64) */
static inline CUtensorMap exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64_encode(
        // Window dataptr, strides
        const void* globalAddress, exo_Sm90_CUtensorMap_5_strides gmem_stride,
        // Tensor size
        exo_Sm90_CUtensorMap_5_dim gmem_dim)
{
    assert(gmem_stride.C_strides[5 - 1] == 1);

    CUtensorMap tensorMap;
    const CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;

    cuuint64_t globalDim[5];
    cuuint64_t allGlobalStrides[5];  // allGlobalStrides[0] unused by CUDA
    cuuint32_t elementStrides[5];

    // We translate from the Exo ordering (leftmost stride is most-significant)
    // to the CUDA ordering (leftmost stride is least-significant).
    for (uint32_t cu_idx = 0; cu_idx < 5; ++cu_idx) {
        const uint32_t C_idx = 5 - 1 - cu_idx;
        globalDim[cu_idx] = gmem_dim.C_dim[C_idx];
        allGlobalStrides[cu_idx] = ((cuuint64_t)gmem_stride.C_strides[C_idx]) * 2;
        elementStrides[cu_idx] = 1;
    }

    cuuint32_t boxDim[5] = { 64, 64, 1, 1, 1 };
    const CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    const CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    const CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    const CUresult result = cuTensorMapEncodeTiled(
            &tensorMap,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            5,
            (void*)globalAddress,
            globalDim,
            &allGlobalStrides[1],  // Cuda presumes least-significant dim is tightly-packed
            boxDim,
            elementStrides,
            interleave,
            swizzle,
            l2Promotion,
            oobFill);
    if (result != 0) {
        fprintf(stderr, "exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64_encode: error %i\n", (int)result);
        assert(0);
    }
    return tensorMap;
}

/* Required by Sm90_tensorMap(128, 1, 1, 128, 64) */
static inline CUtensorMap exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64_encode(
        // Window dataptr, strides
        const void* globalAddress, exo_Sm90_CUtensorMap_4_strides gmem_stride,
        // Tensor size
        exo_Sm90_CUtensorMap_4_dim gmem_dim)
{
    assert(gmem_stride.C_strides[4 - 1] == 1);

    CUtensorMap tensorMap;
    const CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;

    cuuint64_t globalDim[4];
    cuuint64_t allGlobalStrides[4];  // allGlobalStrides[0] unused by CUDA
    cuuint32_t elementStrides[4];

    // We translate from the Exo ordering (leftmost stride is most-significant)
    // to the CUDA ordering (leftmost stride is least-significant).
    for (uint32_t cu_idx = 0; cu_idx < 4; ++cu_idx) {
        const uint32_t C_idx = 4 - 1 - cu_idx;
        globalDim[cu_idx] = gmem_dim.C_dim[C_idx];
        allGlobalStrides[cu_idx] = ((cuuint64_t)gmem_stride.C_strides[C_idx]) * 2;
        elementStrides[cu_idx] = 1;
    }

    cuuint32_t boxDim[4] = { 64, 128, 1, 1 };
    const CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    const CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    const CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    const CUresult result = cuTensorMapEncodeTiled(
            &tensorMap,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            4,
            (void*)globalAddress,
            globalDim,
            &allGlobalStrides[1],  // Cuda presumes least-significant dim is tightly-packed
            boxDim,
            elementStrides,
            interleave,
            swizzle,
            l2Promotion,
            oobFill);
    if (result != 0) {
        fprintf(stderr, "exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64_encode: error %i\n", (int)result);
        assert(0);
    }
    return tensorMap;
}

/* Required by Sm90_tensorMap(0, 1, 1, 1, 64) */
static inline CUtensorMap exo_win_4f32_Sm90_tensorMap_0_1_1_1_64_encode(
        // Window dataptr, strides
        const void* globalAddress, exo_Sm90_CUtensorMap_4_strides gmem_stride,
        // Tensor size
        exo_Sm90_CUtensorMap_4_dim gmem_dim)
{
    assert(gmem_stride.C_strides[4 - 1] == 1);

    CUtensorMap tensorMap;
    const CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;

    cuuint64_t globalDim[4];
    cuuint64_t allGlobalStrides[4];  // allGlobalStrides[0] unused by CUDA
    cuuint32_t elementStrides[4];

    // We translate from the Exo ordering (leftmost stride is most-significant)
    // to the CUDA ordering (leftmost stride is least-significant).
    for (uint32_t cu_idx = 0; cu_idx < 4; ++cu_idx) {
        const uint32_t C_idx = 4 - 1 - cu_idx;
        globalDim[cu_idx] = gmem_dim.C_dim[C_idx];
        allGlobalStrides[cu_idx] = ((cuuint64_t)gmem_stride.C_strides[C_idx]) * 4;
        elementStrides[cu_idx] = 1;
    }

    cuuint32_t boxDim[4] = { 64, 1, 1, 1 };
    const CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    const CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    const CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    const CUresult result = cuTensorMapEncodeTiled(
            &tensorMap,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            4,
            (void*)globalAddress,
            globalDim,
            &allGlobalStrides[1],  // Cuda presumes least-significant dim is tightly-packed
            boxDim,
            elementStrides,
            interleave,
            swizzle,
            l2Promotion,
            oobFill);
    if (result != 0) {
        fprintf(stderr, "exo_win_4f32_Sm90_tensorMap_0_1_1_1_64_encode: error %i\n", (int)result);
        assert(0);
    }
    return tensorMap;
}

// CUDA device function args -- duplicated in .cuh file
struct exo_CudaDeviceArgs0_edited_exo_tk_attn_fwd_Hdim128_causal
{
    int_fast32_t Batch;  // Batch : size
    int_fast32_t KV_Heads;  // KV_Heads : size
    int_fast32_t Groups;  // Groups : size
    int_fast32_t SeqLen;  // SeqLen : size
    CUtensorMap exo_data_o_tm;  //     (Separate window data pointer)
    struct exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64 o_tm;  // o_tm : Window(src_type=bf16[Batch, KV_Heads, Groups, SeqLen, 128], as_tensor=[bf16][Batch, KV_Heads, Groups, SeqLen, 128], src_buf=O, idx='[0:Batch, 0:KV_Heads, 0:Groups, 0:SeqLen, 0:128]') @Sm90_tensorMap(128, 1, 1, 1, 64, 64)
    CUtensorMap exo_data_q_tm;  //     (Separate window data pointer)
    struct exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64 q_tm;  // q_tm : Window(src_type=bf16[Batch, KV_Heads, Groups, SeqLen, 128], as_tensor=[bf16][Batch, KV_Heads, Groups, SeqLen, 128], src_buf=Q, idx='[0:Batch, 0:KV_Heads, 0:Groups, 0:SeqLen, 0:128]') @Sm90_tensorMap(128, 1, 1, 1, 64, 64)
    CUtensorMap exo_data_k_tm;  //     (Separate window data pointer)
    struct exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64 k_tm;  // k_tm : Window(src_type=bf16[Batch, KV_Heads, SeqLen, 128], as_tensor=[bf16][Batch, KV_Heads, SeqLen, 128], src_buf=K, idx='[0:Batch, 0:KV_Heads, 0:SeqLen, 0:128]') @Sm90_tensorMap(128, 1, 1, 128, 64)
    CUtensorMap exo_data_v_tm;  //     (Separate window data pointer)
    struct exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64 v_tm;  // v_tm : Window(src_type=bf16[Batch, KV_Heads, SeqLen, 128], as_tensor=[bf16][Batch, KV_Heads, SeqLen, 128], src_buf=V, idx='[0:Batch, 0:KV_Heads, 0:SeqLen, 0:128]') @Sm90_tensorMap(128, 1, 1, 128, 64)
    CUtensorMap exo_data_lse_tm;  //     (Separate window data pointer)
    struct exo_win_4f32_Sm90_tensorMap_0_1_1_1_64 lse_tm;  // lse_tm : Window(src_type=f32[Batch, KV_Heads, Groups, SeqLen], as_tensor=[f32][Batch, KV_Heads, Groups, SeqLen], src_buf=lse, idx='[0:Batch, 0:KV_Heads, 0:Groups, 0:SeqLen]') @Sm90_tensorMap(0, 1, 1, 1, 64)
    EXO_EXCUT_DEVICE_LOG_MEMBER  // for Exo pytest (exo_excut.h)
};


/* relying on the following instruction..."
Sm90_tk_mma_rmem_row(D,A,B,D=f32, A=bf16, B=bf16, N64=2, K=128, swizzle=128)

*/

/* relying on the following instruction..."
Sm90_tk_mma_row_col(D,A,B,D=f32, A=bf16, B=bf16, N=128, K=64, swizzle=128)

*/

/* relying on the following instruction..."
Sm90_tk_zero_scale_d(D,D=f32, N=128)

*/

/* relying on the following instruction..."
Sm90_tma_load_2d(dst,src,dst=bf16, src=bf16, size0=128, size1=64, smem_box=(1, 1, 128, 64), swizzle=128)

*/

/* relying on the following instruction..."
Sm90_tma_load_2d(dst,src,dst=bf16, src=bf16, size0=64, size1=64, smem_box=(1, 1, 1, 64, 64), swizzle=128)

*/

/* relying on the following instruction..."
Sm90_tma_store_1d(dst,src,dst=f32, src=f32, size0=64, smem_box=(1, 1, 1, 64), swizzle=0)

*/

/* relying on the following instruction..."
Sm90_tma_store_2d(dst,src,dst=bf16, src=bf16, size0=64, size1=64, smem_box=(1, 1, 1, 64, 64), swizzle=128)

*/

/* relying on the following instruction..."
cuda_tk_div_row(dst,src,dst=f32, src=f32, rows=16, cols=128, layout='row')

*/

/* relying on the following instruction..."
cuda_tk_make_causal_neg_infty(row_offset,col_offset,dst,dst=f32, rows=16, cols=128, layout='row')

*/

/* relying on the following instruction..."
cuda_tk_mul_row(dst,src,dst=f32, src=f32, rows=16, cols=128, layout='row')

*/

/* relying on the following instruction..."
cuda_tk_row_max(dst,src,dst=f32, src=f32, rows=16, cols=128, layout='row')

*/

/* relying on the following instruction..."
cuda_tk_row_sum(dst,src,dst=f32, src=f32, rows=16, cols=128, layout='row')

*/

/* relying on the following instruction..."
cuda_tk_store_rs_inner_cols_64(dst,src,dst=bf16, src=f32, rows=16, outer_cols=2)

*/

/* relying on the following instruction..."
cuda_tk_store_vec_rs(dst,src,dst=f32, src=f32, length=16, layout='ortho')

*/

/* relying on the following instruction..."
cuda_tk_sub_row(dst,src,dst=f32, src=f32, rows=16, cols=128, layout='row')

*/

/* relying on the following instruction..."
cuda_tk_tile_add_lhs_scalar(dst,src,dst=f32, src=f32, rows=16, cols=128, layout='row')

*/

/* relying on the following instruction..."
cuda_tk_tile_copy(dst,src,dst=bf16, src=f32, rows=16, cols=128, layout='row')

*/

/* relying on the following instruction..."
cuda_tk_tile_exp2(dst,src,dst=f32, src=f32, rows=16, cols=128, layout='row')

*/

/* relying on the following instruction..."
cuda_tk_tile_mul_3op_scalar(dst,lhs,rhs,dst=f32, lhs=f32, rhs=f32, rows=16, cols=128, layout='row')

*/

/* relying on the following instruction..."
cuda_tk_tile_zero(dst,dst=f32, rows=16, cols=128, layout='row')

*/

/* relying on the following instruction..."
cuda_tk_vec_add_reduce(dst,src,dst=f32, src=f32, length=16, layout='ortho')

*/

/* relying on the following instruction..."
cuda_tk_vec_exp2(dst,src,dst=f32, src=f32, length=16, layout='ortho')

*/

/* relying on the following instruction..."
cuda_tk_vec_log(dst,src,dst=f32, src=f32, length=16, layout='ortho')

*/

/* relying on the following instruction..."
cuda_tk_vec_mul_3op_scalar(dst,lhs,rhs,dst=f32, lhs=f32, rhs=f32, length=16, layout='ortho')

*/

/* relying on the following instruction..."
cuda_tk_vec_mul_lhs(dst,src,dst=f32, src=f32, length=16, layout='ortho')

*/

/* relying on the following instruction..."
cuda_tk_vec_mul_lhs_scalar(dst,src,dst=f32, src=f32, length=16, layout='ortho')

*/

/* relying on the following instruction..."
cuda_tk_vec_neg_infty(dst,dst=f32, length=16, layout='ortho')

*/

/* relying on the following instruction..."
cuda_tk_vec_sub_lhs(dst,src,dst=f32, src=f32, length=16, layout='ortho')

*/

/* relying on the following instruction..."
cuda_tk_vec_zero(dst,dst=f32, length=16, layout='ortho')

*/
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
void edited_exo_tk_attn_fwd_Hdim128_causal( void *ctxt, int_fast32_t Batch, int_fast32_t KV_Heads, int_fast32_t Groups, int_fast32_t SeqLen, exo_bf16* O, float* lse, const exo_bf16* Q, const exo_bf16* K, const exo_bf16* V ) {
  EXO_ASSUME(SeqLen % 16 == 0);
  CUtensorMap exo_data_o_tm = exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64_encode(&O[0], (exo_Sm90_CUtensorMap_5_strides){ {(KV_Heads * Groups * SeqLen * 128), (Groups * SeqLen * 128), (SeqLen * 128), 128, 1} }, (exo_Sm90_CUtensorMap_5_dim){ {Batch, KV_Heads, Groups, SeqLen, 128} });
  struct exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64 o_tm = (exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64) { {0, 0, 0, 0, 0} };
  CUtensorMap exo_data_q_tm = exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64_encode(&Q[0], (exo_Sm90_CUtensorMap_5_strides){ {(KV_Heads * Groups * SeqLen * 128), (Groups * SeqLen * 128), (SeqLen * 128), 128, 1} }, (exo_Sm90_CUtensorMap_5_dim){ {Batch, KV_Heads, Groups, SeqLen, 128} });
  struct exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64 q_tm = (exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64) { {0, 0, 0, 0, 0} };
  CUtensorMap exo_data_k_tm = exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64_encode(&K[0], (exo_Sm90_CUtensorMap_4_strides){ {(KV_Heads * SeqLen * 128), (SeqLen * 128), 128, 1} }, (exo_Sm90_CUtensorMap_4_dim){ {Batch, KV_Heads, SeqLen, 128} });
  struct exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64 k_tm = (exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64) { {0, 0, 0, 0} };
  CUtensorMap exo_data_v_tm = exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64_encode(&V[0], (exo_Sm90_CUtensorMap_4_strides){ {(KV_Heads * SeqLen * 128), (SeqLen * 128), 128, 1} }, (exo_Sm90_CUtensorMap_4_dim){ {Batch, KV_Heads, SeqLen, 128} });
  struct exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64 v_tm = (exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64) { {0, 0, 0, 0} };
  CUtensorMap exo_data_lse_tm = exo_win_4f32_Sm90_tensorMap_0_1_1_1_64_encode(&lse[0], (exo_Sm90_CUtensorMap_4_strides){ {(KV_Heads * Groups * SeqLen), (Groups * SeqLen), SeqLen, 1} }, (exo_Sm90_CUtensorMap_4_dim){ {Batch, KV_Heads, Groups, SeqLen} });
  struct exo_win_4f32_Sm90_tensorMap_0_1_1_1_64 lse_tm = (exo_win_4f32_Sm90_tensorMap_0_1_1_1_64) { {0, 0, 0, 0} };
  {
    struct exo_CudaDeviceArgs0_edited_exo_tk_attn_fwd_Hdim128_causal exo_deviceArgs = {
      Batch, KV_Heads, Groups, SeqLen, exo_data_o_tm, (exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64) { {o_tm.C_offsets[0], o_tm.C_offsets[1], o_tm.C_offsets[2], o_tm.C_offsets[3], o_tm.C_offsets[4]} }, exo_data_q_tm, (exo_win_5bf16_Sm90_tensorMap_128_1_1_1_64_64) { {q_tm.C_offsets[0], q_tm.C_offsets[1], q_tm.C_offsets[2], q_tm.C_offsets[3], q_tm.C_offsets[4]} }, exo_data_k_tm, (exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64) { {k_tm.C_offsets[0], k_tm.C_offsets[1], k_tm.C_offsets[2], k_tm.C_offsets[3]} }, exo_data_v_tm, (exo_win_4bf16_Sm90_tensorMap_128_1_1_128_64) { {v_tm.C_offsets[0], v_tm.C_offsets[1], v_tm.C_offsets[2], v_tm.C_offsets[3]} }, exo_data_lse_tm, (exo_win_4f32_Sm90_tensorMap_0_1_1_1_64) { {lse_tm.C_offsets[0], lse_tm.C_offsets[1], lse_tm.C_offsets[2], lse_tm.C_offsets[3]} }, exo_excut_get_device_log()
    };
    exo_cudaLaunch0_edited_exo_tk_attn_fwd_Hdim128_causal(exo_cudaStream, &exo_deviceArgs);
  }
}

