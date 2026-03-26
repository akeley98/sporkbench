#pragma once

#define EDIT_NO_PERSISTENT 1
#define EDIT_TASK_INDEX_32 1
#define EDIT_WGMMA_DESC 1
#define EDIT_MBARRIER 1
#define EDIT_SMART_LOOP_BOUNDS 1

#if EDIT_MBARRIER
#if !EDIT_NO_PERSISTENT
#error "mbarrier changes won't work with persistent kernel"
#endif
#endif

#include "exocc_Sm90a_edited_tk_attn_fwd_causal.h"
#if EXO_EXCUT_bENABLE_LOG
#include "exocc_Sm90a_edited_tk_attn_fwd_causal.excut_str_table"
#endif
/* Required by Sm90_tensorMap(0, 1, 1, 1, 64) */
/* Required by Sm90_tensorMap(128, 1, 1, 1, 64, 64) */
/* Required by Sm90_tensorMap(128, 1, 1, 128, 64) */
#include "cuda.h"
/* Required by CudaTkWarpTile(16, 128, 'row') */
/* Required by CudaTkWarpVec(16, 'ortho') */
/* Required by Sm90_TkRmemTileA(128,) */
/* Required by Sm90_TkRmemTileD(128,) */
/* Required by Sm90_tk_mma_rmem_row(D,A,B,D=f32, A=bf16, B=bf16, N64=2, K=128, swizzle=128) */
/* Required by Sm90_tk_mma_row_col(D,A,B,D=f32, A=bf16, B=bf16, N=128, K=64, swizzle=128) */
/* Required by cuda_tk_div_row(dst,src,dst=f32, src=f32, rows=16, cols=128, layout='row') */
/* Required by cuda_tk_make_causal_neg_infty(row_offset,col_offset,dst,dst=f32, rows=16, cols=128, layout='row') */
/* Required by cuda_tk_mul_row(dst,src,dst=f32, src=f32, rows=16, cols=128, layout='row') */
/* Required by cuda_tk_row_max(dst,src,dst=f32, src=f32, rows=16, cols=128, layout='row') */
/* Required by cuda_tk_row_sum(dst,src,dst=f32, src=f32, rows=16, cols=128, layout='row') */
/* Required by cuda_tk_store_rs_inner_cols_64(dst,src,dst=bf16, src=f32, rows=16, outer_cols=2) */
/* Required by cuda_tk_store_vec_rs(dst,src,dst=f32, src=f32, length=16, layout='ortho') */
/* Required by cuda_tk_sub_row(dst,src,dst=f32, src=f32, rows=16, cols=128, layout='row') */
/* Required by cuda_tk_tile_add_lhs_scalar(dst,src,dst=f32, src=f32, rows=16, cols=128, layout='row') */
/* Required by cuda_tk_tile_copy(dst,src,dst=bf16, src=f32, rows=16, cols=128, layout='row') */
/* Required by cuda_tk_tile_exp2(dst,src,dst=f32, src=f32, rows=16, cols=128, layout='row') */
/* Required by cuda_tk_tile_mul_3op_scalar(dst,lhs,rhs,dst=f32, lhs=f32, rhs=f32, rows=16, cols=128, layout='row') */
/* Required by cuda_tk_tile_zero(dst,dst=f32, rows=16, cols=128, layout='row') */
/* Required by cuda_tk_vec_add_reduce(dst,src,dst=f32, src=f32, length=16, layout='ortho') */
/* Required by cuda_tk_vec_exp2(dst,src,dst=f32, src=f32, length=16, layout='ortho') */
/* Required by cuda_tk_vec_log(dst,src,dst=f32, src=f32, length=16, layout='ortho') */
/* Required by cuda_tk_vec_mul_3op_scalar(dst,lhs,rhs,dst=f32, lhs=f32, rhs=f32, length=16, layout='ortho') */
/* Required by cuda_tk_vec_mul_lhs(dst,src,dst=f32, src=f32, length=16, layout='ortho') */
/* Required by cuda_tk_vec_mul_lhs_scalar(dst,src,dst=f32, src=f32, length=16, layout='ortho') */
/* Required by cuda_tk_vec_neg_infty(dst,dst=f32, length=16, layout='ortho') */
/* Required by cuda_tk_vec_sub_lhs(dst,src,dst=f32, src=f32, length=16, layout='ortho') */
/* Required by cuda_tk_vec_zero(dst,dst=f32, length=16, layout='ortho') */
#include <kittens.cuh>
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

namespace exo_CudaUtil_exocc_Sm90a_edited_tk_attn_fwd_causal {
namespace exo_CudaUtil = ::exo_CudaUtil_exocc_Sm90a_edited_tk_attn_fwd_causal;
/* Required by Sm90_tk_mma_rmem_row(D,A,B,D=f32, A=bf16, B=bf16, N64=2, K=128, swizzle=128) */
/* Required by Sm90_tk_mma_row_col(D,A,B,D=f32, A=bf16, B=bf16, N=128, K=64, swizzle=128) */

EXO_CUDA_INLINE uint64_t exo_Sm90_matrix_descriptor_encode(uint32_t val)
{
    return (val & 0x3FFFF) >> 4;
}

template <typename SwizzledElement>
EXO_CUDA_INLINE
uint64_t exo_Sm90_smem_descriptor(SwizzledElement* ptr, uint32_t leading_stride_elements, uint32_t stride_stride_elements)
{
    uint32_t element_size = sizeof(SwizzledElement);
    return (
        exo_Sm90_matrix_descriptor_encode(exo_smemU32(ptr))
      | exo_Sm90_matrix_descriptor_encode(leading_stride_elements * element_size) << 16u
      | exo_Sm90_matrix_descriptor_encode(stride_stride_elements * element_size) << 32u
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 100
      | uint64_t(1) << 46  // Bit-field 46-48, Fixed constant value of 0b001 on Blackwell (?!!?)
#endif
      | uint64_t(SwizzledElement::get_swizzle_bits()) << 62
  );
}

/* Required by Sm90_tma_load_2d(dst,src,dst=bf16, src=bf16, size0=128, size1=64, smem_box=(1, 1, 128, 64), swizzle=128) */
/* Required by Sm90_tma_load_2d(dst,src,dst=bf16, src=bf16, size0=64, size1=64, smem_box=(1, 1, 1, 64, 64), swizzle=128) */
template <typename WindowOffsets>
EXO_CUDA_INLINE void
exo_Sm90_tma_to_smem(
        void* dst, const CUtensorMap& tensorMap, WindowOffsets window,
        uint32_t exo_tma_mbarrier, uint32_t expect_tx)
{
    constexpr auto rank = sizeof(window.C_offsets) / sizeof(window.C_offsets[0]);
    static_assert(rank >= 1 && rank <= 5);
    // cute::elect_one_sync
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "     elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "     mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFF));
    if (pred) {
        asm("mbarrier.expect_tx.shared::cta.b64 [%0], %1;" :: "r"(exo_tma_mbarrier), "r"(expect_tx));
        if constexpr (rank == 1) {
            asm volatile(
            "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2}], [%3], %4;"
            :
            : "r"(exo_smemU32(dst)), "l"(&tensorMap), "r"(window.C_offsets[0]),
              "r"(exo_tma_mbarrier), "n"(1152921504606846976)
            : "memory");
        }
        if constexpr (rank == 2) {
            asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3}], [%4], %5;"
            :
            : "r"(exo_smemU32(dst)), "l"(&tensorMap), "r"(window.C_offsets[1]), "r"(window.C_offsets[0]),
              "r"(exo_tma_mbarrier), "n"(1152921504606846976)
            : "memory");
        }
        if constexpr (rank == 3) {
            asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
            :
            : "r"(exo_smemU32(dst)), "l"(&tensorMap), "r"(window.C_offsets[2]), "r"(window.C_offsets[1]), "r"(window.C_offsets[0]),
              "r"(exo_tma_mbarrier), "n"(1152921504606846976)
            : "memory");
        }
        if constexpr (rank == 4) {
            asm volatile(
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
            :
            : "r"(exo_smemU32(dst)), "l"(&tensorMap), "r"(window.C_offsets[3]), "r"(window.C_offsets[2]), "r"(window.C_offsets[1]), "r"(window.C_offsets[0]),
              "r"(exo_tma_mbarrier), "n"(1152921504606846976)
            : "memory");
        }
        if constexpr (rank == 5) {
            asm volatile(
            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
            :
            : "r"(exo_smemU32(dst)), "l"(&tensorMap), "r"(window.C_offsets[4]), "r"(window.C_offsets[3]), "r"(window.C_offsets[2]), "r"(window.C_offsets[1]), "r"(window.C_offsets[0]),
              "r"(exo_tma_mbarrier), "n"(1152921504606846976)
            : "memory");
        }
    }
}
/* Required by Sm90_tma_store_1d(dst,src,dst=f32, src=f32, size0=64, smem_box=(1, 1, 1, 64), swizzle=0) */
/* Required by Sm90_tma_store_2d(dst,src,dst=bf16, src=bf16, size0=64, size1=64, smem_box=(1, 1, 1, 64, 64), swizzle=128) */
template <typename WindowOffsets>
EXO_CUDA_INLINE void
exo_Sm90_tma_to_gmem(const CUtensorMap& tensorMap, WindowOffsets window, const void* src)
{
    constexpr auto rank = sizeof(window.C_offsets) / sizeof(window.C_offsets[0]);
    static_assert(rank >= 1 && rank <= 5);
    // cute::elect_one_sync
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "     elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "     mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFF));
    if (pred) {
        if constexpr (rank == 1) {
            asm volatile(
                "cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group"
                " [%0, {%1}], [%2];"
                :
                : "l"(&tensorMap),
                  "r"(window.C_offsets[0]),
                  "r"(exo_smemU32(src))
                : "memory");
        }
        if constexpr (rank == 2) {
            asm volatile(
                "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
                " [%0, {%1, %2}], [%3];"
                :
                : "l"(&tensorMap),
                  "r"(window.C_offsets[1]), "r"(window.C_offsets[0]),
                  "r"(exo_smemU32(src))
                : "memory");
        }
        if constexpr (rank == 3) {
            asm volatile(
                "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
                " [%0, {%1, %2, %3}], [%4];"
                :
                : "l"(&tensorMap),
                  "r"(window.C_offsets[2]), "r"(window.C_offsets[1]), "r"(window.C_offsets[0]),
                  "r"(exo_smemU32(src))
                : "memory");
        }
        if constexpr (rank == 4) {
            asm volatile(
                "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
                " [%0, {%1, %2, %3, %4}], [%5];"
                :
                : "l"(&tensorMap),
                  "r"(window.C_offsets[3]), "r"(window.C_offsets[2]), "r"(window.C_offsets[1]), "r"(window.C_offsets[0]),
                  "r"(exo_smemU32(src))
                : "memory");
        }
        if constexpr (rank == 5) {
            asm volatile(
                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
                " [%0, {%1, %2, %3, %4, %5}], [%6];"
                :
                : "l"(&tensorMap),
                  "r"(window.C_offsets[4]), "r"(window.C_offsets[3]), "r"(window.C_offsets[2]), "r"(window.C_offsets[1]), "r"(window.C_offsets[0]),
                  "r"(exo_smemU32(src))
                : "memory");
        }
    }
}
/* Required by cuda_tk_store_vec_rs(dst,src,dst=f32, src=f32, length=16, layout='ortho') */
template <int _length, typename _T>
EXO_CUDA_INLINE
::kittens::sv<_T, _length>&
exo_tk_cast_sv(const _T* smem_ptr)
{
    using vec_t = ::kittens::sv<_T, _length>;
    // Crazy: Kittens rounds up struct size to 128 bytes.
    // We have to hope the store op using this cast
    // doesn't write past the end.
    // static_assert(sizeof(vec_t) == _length * sizeof(_T));
    return *reinterpret_cast<vec_t*>(const_cast<_T*>(smem_ptr));
}

}  // end namespace exo_CudaUtil_exocc_Sm90a_edited_tk_attn_fwd_causal
// CUDA device function args -- duplicated in .c file
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

// We need this inline namespace to avoid ODR problems in pytest.
inline namespace exo_CudaInline_exocc_Sm90a_edited_tk_attn_fwd_causal {
struct exo_Cuda0_edited_exo_tk_attn_fwd_Hdim128_causal
{
  using exo_DeviceArgs = exo_CudaDeviceArgs0_edited_exo_tk_attn_fwd_Hdim128_causal;

  static constexpr uint32_t exo_blockDim = 512;
  static constexpr uint32_t exo_clusterDim = 1;

  static constexpr unsigned exo_smemBytes = 181120;
  static constexpr unsigned exo_smemOffset0_qo_smem = 128;  // 49152-byte allocation
  static constexpr unsigned exo_smemOffset1_k_smem = 49280;  // 65536-byte allocation
  static constexpr unsigned exo_smemOffset2_v_smem = 114816;  // 65536-byte allocation
  static constexpr unsigned exo_smemOffset3_lse_smem = 180352;  // 768-byte allocation
  static constexpr unsigned exo_smemOffset4_v_consumed = 0;  // 16-byte allocation
  static constexpr unsigned exo_smemOffset5_k_consumed = 16;  // 16-byte allocation
  static constexpr unsigned exo_smemOffset6_q_tmp_barrier = 32;  // 8-byte allocation
  static constexpr unsigned exo_smemOffset7_v_produced = 40;  // 16-byte allocation
  static constexpr unsigned exo_smemOffset8_k_produced = 56;  // 16-byte allocation
  static constexpr unsigned exo_smemOffset9_q_produced = 72;  // 8-byte allocation

#if EDIT_TASK_INDEX_32
  using task_index_t = int32_t;
#else
  using task_index_t = int64_t;
#endif

  struct exo_Task
  {
    task_index_t batch;
    task_index_t kv_head;
    task_index_t group;
    task_index_t qo_task;
  };

  struct exo_TaskGenerator
  {
    uint32_t exo_taskIndex;
    uint32_t exo_numClusters;
    uint32_t exo_taskCount;
    int_fast32_t exo_cudaTasksLo_batch;
    uint32_t exo_cudaTasksNum_batch;
    int_fast32_t exo_cudaTasksLo_kv_head;
    uint32_t exo_cudaTasksNum_kv_head;
    int_fast32_t exo_cudaTasksLo_group;
    uint32_t exo_cudaTasksNum_group;
    int_fast32_t exo_cudaTasksLo_qo_task;
    uint32_t exo_cudaTasksNum_qo_task;
    EXO_CUDA_INLINE exo_TaskGenerator(
        uint32_t cluster_index, uint32_t num_clusters,
        int_fast32_t _exo_cudaTasksLo_batch, int_fast32_t _exo_cudaTasksHi_batch,
        int_fast32_t _exo_cudaTasksLo_kv_head, int_fast32_t _exo_cudaTasksHi_kv_head,
        int_fast32_t _exo_cudaTasksLo_group, int_fast32_t _exo_cudaTasksHi_group,
        int_fast32_t _exo_cudaTasksLo_qo_task, int_fast32_t _exo_cudaTasksHi_qo_task,
        const exo_DeviceArgs&)
    {
      exo_taskIndex = cluster_index;
      exo_numClusters = num_clusters;
      exo_taskCount = 1;
      exo_cudaTasksLo_batch = _exo_cudaTasksLo_batch;
      exo_cudaTasksNum_batch = static_cast<uint32_t>(_exo_cudaTasksHi_batch - _exo_cudaTasksLo_batch);
      exo_taskCount *= exo_cudaTasksNum_batch;
      exo_cudaTasksLo_kv_head = _exo_cudaTasksLo_kv_head;
      exo_cudaTasksNum_kv_head = static_cast<uint32_t>(_exo_cudaTasksHi_kv_head - _exo_cudaTasksLo_kv_head);
      exo_taskCount *= exo_cudaTasksNum_kv_head;
      exo_cudaTasksLo_group = _exo_cudaTasksLo_group;
      exo_cudaTasksNum_group = static_cast<uint32_t>(_exo_cudaTasksHi_group - _exo_cudaTasksLo_group);
      exo_taskCount *= exo_cudaTasksNum_group;
      exo_cudaTasksLo_qo_task = _exo_cudaTasksLo_qo_task;
      exo_cudaTasksNum_qo_task = static_cast<uint32_t>(_exo_cudaTasksHi_qo_task - _exo_cudaTasksLo_qo_task);
      exo_taskCount *= exo_cudaTasksNum_qo_task;
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
      exo_task.qo_task = task_index_t(exo_cudaTasksLo_qo_task + static_cast<int_fast32_t>(exo_tmp % exo_cudaTasksNum_qo_task));
      exo_tmp /= exo_cudaTasksNum_qo_task;
      exo_task.group = task_index_t(exo_cudaTasksLo_group + static_cast<int_fast32_t>(exo_tmp % exo_cudaTasksNum_group));
      exo_tmp /= exo_cudaTasksNum_group;
      exo_task.kv_head = task_index_t(exo_cudaTasksLo_kv_head + static_cast<int_fast32_t>(exo_tmp % exo_cudaTasksNum_kv_head));
      exo_tmp /= exo_cudaTasksNum_kv_head;
      exo_task.batch = task_index_t(exo_cudaTasksLo_batch + static_cast<int_fast32_t>(exo_tmp % exo_cudaTasksNum_batch));
      exo_tmp /= exo_cudaTasksNum_batch;
      return exo_task;
    }
  };

  struct exo_SyncState
  {
#if EDIT_MBARRIER
  // static constexpr unsigned exo_smemOffset4_v_consumed = 0;  // 16-byte allocation
  // static constexpr unsigned exo_smemOffset5_k_consumed = 16;  // 16-byte allocation
  // static constexpr unsigned exo_smemOffset6_q_tmp_barrier = 32;  // 8-byte allocation
  // static constexpr unsigned exo_smemOffset7_v_produced = 40;  // 16-byte allocation
  // static constexpr unsigned exo_smemOffset8_k_produced = 56;  // 16-byte allocation
  // static constexpr unsigned exo_smemOffset9_q_produced = 72;  // 8-byte allocation
    EXO_CUDA_INLINE kittens::semaphore* get_v_consumed(char* exo_smem)
    {
        return reinterpret_cast<kittens::semaphore*>(exo_smem + exo_smemOffset4_v_consumed);
    }
    EXO_CUDA_INLINE kittens::semaphore* get_k_consumed(char* exo_smem)
    {
        return reinterpret_cast<kittens::semaphore*>(exo_smem + exo_smemOffset5_k_consumed);
    }
    // EXO_CUDA_INLINE kittens::semaphore& get_q_tmp_barrier(char* exo_smem)
    // {
    //     return reinterpret_cast<kittens::semaphore&>(exo_smem[exo_smemOffset6_q_tmp_barrier]);
    // }
    EXO_CUDA_INLINE kittens::semaphore* get_v_produced(char* exo_smem)
    {
        return reinterpret_cast<kittens::semaphore*>(exo_smem + exo_smemOffset7_v_produced);
    }
    EXO_CUDA_INLINE kittens::semaphore* get_k_produced(char* exo_smem)
    {
        return reinterpret_cast<kittens::semaphore*>(exo_smem + exo_smemOffset8_k_produced);
    }
    EXO_CUDA_INLINE kittens::semaphore& get_q_produced(char* exo_smem)
    {
        return reinterpret_cast<kittens::semaphore&>(exo_smem[exo_smemOffset9_q_produced]);
    }
#else
    // v_consumed: barrier @ CudaMbarrier, ring=2, slice_count=1
    // num_per_cta=2; arrive_count=384
    unsigned ArriveIdx0_v_consumed = 0;
    EXO_CUDA_INLINE uint32_t Arrive0_v_consumed(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice, bool enable) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + exo_smemOffset4_v_consumed + 8*(slice * 2 + ArriveIdx0_v_consumed));
      if (enable) {
        asm volatile(
          "// Arrive0_v_consumed\n\t"
          "mbarrier.arrive.shared::cta.b64 _, [%0];"
            :
            :"r"(mbarrier_u32)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_arrive_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        // Advance ring buffer state
        ArriveIdx0_v_consumed = ArriveIdx0_v_consumed == 1 ? 0 : ArriveIdx0_v_consumed + 1;
      }
      return mbarrier_u32;
    }
    unsigned AwaitIdx0_v_consumed = 0;
    unsigned Parity0_v_consumed = 0;
    unsigned Skips0_v_consumed = 0;
    EXO_CUDA_INLINE void Await0_v_consumed(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice, int initial_skips = 0) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + exo_smemOffset4_v_consumed + 8*(slice * 2 + AwaitIdx0_v_consumed));
      const bool enable = Skips0_v_consumed >= initial_skips;
      if (enable) {
    #if __CUDA_ARCH__ < 900
        asm volatile(
          "{\n\t"
          "// Await0_v_consumed\n\t"
          ".reg.pred P1;\n\t"
          "EXO_BEFORE_WAIT:\n\t"
          "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
          "@P1 bra.uni EXO_WAIT_DONE;\n\t"
          "bra.uni EXO_BEFORE_WAIT;\n\t"
          "EXO_WAIT_DONE:\n\t"
          "}"
            :
            :"r"(mbarrier_u32),
            "r"(1u & Parity0_v_consumed >> AwaitIdx0_v_consumed)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_test_wait_parity_acquire_cta_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(1u & Parity0_v_consumed >> AwaitIdx0_v_consumed));
    #else
        asm volatile(
          "{\n\t"
          "// Await0_v_consumed\n\t"
          ".reg.pred P1;\n\t"
          "EXO_BEFORE_WAIT:\n\t"
          "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
          "@P1 bra.uni EXO_WAIT_DONE;\n\t"
          "bra.uni EXO_BEFORE_WAIT;\n\t"
          "EXO_WAIT_DONE:\n\t"
          "}"
            :
            :"r"(mbarrier_u32),
            "r"(1u & Parity0_v_consumed >> AwaitIdx0_v_consumed)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_try_wait_parity_acquire_cta_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(1u & Parity0_v_consumed >> AwaitIdx0_v_consumed));
    #endif
        // Flip parity
        Parity0_v_consumed ^= 1u << AwaitIdx0_v_consumed;
        // Advance ring buffer state
        AwaitIdx0_v_consumed = AwaitIdx0_v_consumed == 1 ? 0 : AwaitIdx0_v_consumed + 1;
      }
      else {
        // Await(v_consumed) returns without waiting for mbarrier first <initial_skips> times
        Skips0_v_consumed++;
      }
    }
    // k_consumed: barrier @ CudaMbarrier, ring=2, slice_count=1
    // num_per_cta=2; arrive_count=384
    unsigned ArriveIdx0_k_consumed = 0;
    EXO_CUDA_INLINE uint32_t Arrive0_k_consumed(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice, bool enable) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + exo_smemOffset5_k_consumed + 8*(slice * 2 + ArriveIdx0_k_consumed));
      if (enable) {
        asm volatile(
          "// Arrive0_k_consumed\n\t"
          "mbarrier.arrive.shared::cta.b64 _, [%0];"
            :
            :"r"(mbarrier_u32)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_arrive_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        // Advance ring buffer state
        ArriveIdx0_k_consumed = ArriveIdx0_k_consumed == 1 ? 0 : ArriveIdx0_k_consumed + 1;
      }
      return mbarrier_u32;
    }
    unsigned AwaitIdx0_k_consumed = 0;
    unsigned Parity0_k_consumed = 0;
    unsigned Skips0_k_consumed = 0;
    EXO_CUDA_INLINE void Await0_k_consumed(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice, int initial_skips = 0) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + exo_smemOffset5_k_consumed + 8*(slice * 2 + AwaitIdx0_k_consumed));
      const bool enable = Skips0_k_consumed >= initial_skips;
      if (enable) {
    #if __CUDA_ARCH__ < 900
        asm volatile(
          "{\n\t"
          "// Await0_k_consumed\n\t"
          ".reg.pred P1;\n\t"
          "EXO_BEFORE_WAIT:\n\t"
          "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
          "@P1 bra.uni EXO_WAIT_DONE;\n\t"
          "bra.uni EXO_BEFORE_WAIT;\n\t"
          "EXO_WAIT_DONE:\n\t"
          "}"
            :
            :"r"(mbarrier_u32),
            "r"(1u & Parity0_k_consumed >> AwaitIdx0_k_consumed)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_test_wait_parity_acquire_cta_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(1u & Parity0_k_consumed >> AwaitIdx0_k_consumed));
    #else
        asm volatile(
          "{\n\t"
          "// Await0_k_consumed\n\t"
          ".reg.pred P1;\n\t"
          "EXO_BEFORE_WAIT:\n\t"
          "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
          "@P1 bra.uni EXO_WAIT_DONE;\n\t"
          "bra.uni EXO_BEFORE_WAIT;\n\t"
          "EXO_WAIT_DONE:\n\t"
          "}"
            :
            :"r"(mbarrier_u32),
            "r"(1u & Parity0_k_consumed >> AwaitIdx0_k_consumed)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_try_wait_parity_acquire_cta_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(1u & Parity0_k_consumed >> AwaitIdx0_k_consumed));
    #endif
        // Flip parity
        Parity0_k_consumed ^= 1u << AwaitIdx0_k_consumed;
        // Advance ring buffer state
        AwaitIdx0_k_consumed = AwaitIdx0_k_consumed == 1 ? 0 : AwaitIdx0_k_consumed + 1;
      }
      else {
        // Await(k_consumed) returns without waiting for mbarrier first <initial_skips> times
        Skips0_k_consumed++;
      }
    }
    // q_tmp_barrier: barrier @ CudaMbarrier, ring=1, slice_count=1
    // num_per_cta=1; arrive_count=384
    static constexpr unsigned ArriveIdx0_q_tmp_barrier = 0;  // Trivial size-1 ring buffer
    EXO_CUDA_INLINE uint32_t Arrive0_q_tmp_barrier(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice, bool enable) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + exo_smemOffset6_q_tmp_barrier + 8*(slice * 1 + ArriveIdx0_q_tmp_barrier));
      if (enable) {
        asm volatile(
          "// Arrive0_q_tmp_barrier\n\t"
          "mbarrier.arrive.shared::cta.b64 _, [%0];"
            :
            :"r"(mbarrier_u32)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_arrive_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
      }
      return mbarrier_u32;
    }
    static constexpr unsigned AwaitIdx0_q_tmp_barrier = 0;  // Trivial size-1 ring buffer
    unsigned Parity0_q_tmp_barrier = 0;
    unsigned Skips0_q_tmp_barrier = 0;
    EXO_CUDA_INLINE void Await0_q_tmp_barrier(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice, int initial_skips = 0) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + exo_smemOffset6_q_tmp_barrier + 8*(slice * 1 + AwaitIdx0_q_tmp_barrier));
      const bool enable = Skips0_q_tmp_barrier >= initial_skips;
      if (enable) {
    #if __CUDA_ARCH__ < 900
        asm volatile(
          "{\n\t"
          "// Await0_q_tmp_barrier\n\t"
          ".reg.pred P1;\n\t"
          "EXO_BEFORE_WAIT:\n\t"
          "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
          "@P1 bra.uni EXO_WAIT_DONE;\n\t"
          "bra.uni EXO_BEFORE_WAIT;\n\t"
          "EXO_WAIT_DONE:\n\t"
          "}"
            :
            :"r"(mbarrier_u32),
            "r"(1u & Parity0_q_tmp_barrier >> AwaitIdx0_q_tmp_barrier)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_test_wait_parity_acquire_cta_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(1u & Parity0_q_tmp_barrier >> AwaitIdx0_q_tmp_barrier));
    #else
        asm volatile(
          "{\n\t"
          "// Await0_q_tmp_barrier\n\t"
          ".reg.pred P1;\n\t"
          "EXO_BEFORE_WAIT:\n\t"
          "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
          "@P1 bra.uni EXO_WAIT_DONE;\n\t"
          "bra.uni EXO_BEFORE_WAIT;\n\t"
          "EXO_WAIT_DONE:\n\t"
          "}"
            :
            :"r"(mbarrier_u32),
            "r"(1u & Parity0_q_tmp_barrier >> AwaitIdx0_q_tmp_barrier)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_try_wait_parity_acquire_cta_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(1u & Parity0_q_tmp_barrier >> AwaitIdx0_q_tmp_barrier));
    #endif
        // Flip parity
        Parity0_q_tmp_barrier ^= 1u << AwaitIdx0_q_tmp_barrier;
      }
      else {
        // Await(q_tmp_barrier) returns without waiting for mbarrier first <initial_skips> times
        Skips0_q_tmp_barrier++;
      }
    }
    // v_produced: barrier @ CudaMbarrier, ring=2, slice_count=1
    // num_per_cta=2; arrive_count=32
    unsigned ArriveIdx0_v_produced = 0;
    EXO_CUDA_INLINE uint32_t Arrive0_v_produced(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice, bool enable) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + exo_smemOffset7_v_produced + 8*(slice * 2 + ArriveIdx0_v_produced));
      if (enable) {
        asm volatile(
          "// Arrive0_v_produced\n\t"
          "mbarrier.arrive.shared::cta.b64 _, [%0];"
            :
            :"r"(mbarrier_u32)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_arrive_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        // Advance ring buffer state
        ArriveIdx0_v_produced = ArriveIdx0_v_produced == 1 ? 0 : ArriveIdx0_v_produced + 1;
      }
      return mbarrier_u32;
    }
    unsigned AwaitIdx0_v_produced = 0;
    unsigned Parity0_v_produced = 0;
    EXO_CUDA_INLINE void Await0_v_produced(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + exo_smemOffset7_v_produced + 8*(slice * 2 + AwaitIdx0_v_produced));
      const bool enable = true;
      if (enable) {
    #if __CUDA_ARCH__ < 900
        asm volatile(
          "{\n\t"
          "// Await0_v_produced\n\t"
          ".reg.pred P1;\n\t"
          "EXO_BEFORE_WAIT:\n\t"
          "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
          "@P1 bra.uni EXO_WAIT_DONE;\n\t"
          "bra.uni EXO_BEFORE_WAIT;\n\t"
          "EXO_WAIT_DONE:\n\t"
          "}"
            :
            :"r"(mbarrier_u32),
            "r"(1u & Parity0_v_produced >> AwaitIdx0_v_produced)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_test_wait_parity_acquire_cta_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(1u & Parity0_v_produced >> AwaitIdx0_v_produced));
    #else
        asm volatile(
          "{\n\t"
          "// Await0_v_produced\n\t"
          ".reg.pred P1;\n\t"
          "EXO_BEFORE_WAIT:\n\t"
          "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
          "@P1 bra.uni EXO_WAIT_DONE;\n\t"
          "bra.uni EXO_BEFORE_WAIT;\n\t"
          "EXO_WAIT_DONE:\n\t"
          "}"
            :
            :"r"(mbarrier_u32),
            "r"(1u & Parity0_v_produced >> AwaitIdx0_v_produced)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_try_wait_parity_acquire_cta_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(1u & Parity0_v_produced >> AwaitIdx0_v_produced));
    #endif
        // Flip parity
        Parity0_v_produced ^= 1u << AwaitIdx0_v_produced;
        // Advance ring buffer state
        AwaitIdx0_v_produced = AwaitIdx0_v_produced == 1 ? 0 : AwaitIdx0_v_produced + 1;
      }
    }
    // k_produced: barrier @ CudaMbarrier, ring=2, slice_count=1
    // num_per_cta=2; arrive_count=32
    unsigned ArriveIdx0_k_produced = 0;
    EXO_CUDA_INLINE uint32_t Arrive0_k_produced(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice, bool enable) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + exo_smemOffset8_k_produced + 8*(slice * 2 + ArriveIdx0_k_produced));
      if (enable) {
        asm volatile(
          "// Arrive0_k_produced\n\t"
          "mbarrier.arrive.shared::cta.b64 _, [%0];"  // XXX
          // "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], 1;"
            :
            :"r"(mbarrier_u32)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_arrive_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        // Advance ring buffer state
        ArriveIdx0_k_produced = ArriveIdx0_k_produced == 1 ? 0 : ArriveIdx0_k_produced + 1;
      }
      return mbarrier_u32;
    }
    unsigned AwaitIdx0_k_produced = 0;
    unsigned Parity0_k_produced = 0;
    EXO_CUDA_INLINE void Await0_k_produced(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + exo_smemOffset8_k_produced + 8*(slice * 2 + AwaitIdx0_k_produced));
      const bool enable = true;
      if (enable) {
    #if __CUDA_ARCH__ < 900
        asm volatile(
          "{\n\t"
          "// Await0_k_produced\n\t"
          ".reg.pred P1;\n\t"
          "EXO_BEFORE_WAIT:\n\t"
          "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
          "@P1 bra.uni EXO_WAIT_DONE;\n\t"
          "bra.uni EXO_BEFORE_WAIT;\n\t"
          "EXO_WAIT_DONE:\n\t"
          "}"
            :
            :"r"(mbarrier_u32),
            "r"(1u & Parity0_k_produced >> AwaitIdx0_k_produced)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_test_wait_parity_acquire_cta_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(1u & Parity0_k_produced >> AwaitIdx0_k_produced));
    #else
        asm volatile(
          "{\n\t"
          "// Await0_k_produced\n\t"
          ".reg.pred P1;\n\t"
          "EXO_BEFORE_WAIT:\n\t"
          "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
          "@P1 bra.uni EXO_WAIT_DONE;\n\t"
          "bra.uni EXO_BEFORE_WAIT;\n\t"
          "EXO_WAIT_DONE:\n\t"
          "}"
            :
            :"r"(mbarrier_u32),
            "r"(1u & Parity0_k_produced >> AwaitIdx0_k_produced)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_try_wait_parity_acquire_cta_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(1u & Parity0_k_produced >> AwaitIdx0_k_produced));
    #endif
        // Flip parity
        Parity0_k_produced ^= 1u << AwaitIdx0_k_produced;
        // Advance ring buffer state
        AwaitIdx0_k_produced = AwaitIdx0_k_produced == 1 ? 0 : AwaitIdx0_k_produced + 1;
      }
    }
    // q_produced: barrier @ CudaMbarrier, ring=1, slice_count=1
    // num_per_cta=1; arrive_count=32
    static constexpr unsigned ArriveIdx0_q_produced = 0;  // Trivial size-1 ring buffer
    EXO_CUDA_INLINE uint32_t Arrive0_q_produced(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice, bool enable) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + exo_smemOffset9_q_produced + 8*(slice * 1 + ArriveIdx0_q_produced));
      if (enable) {
        asm volatile(
          "// Arrive0_q_produced\n\t"
          "mbarrier.arrive.shared::cta.b64 _, [%0];"
            :
            :"r"(mbarrier_u32)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_arrive_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
      }
      return mbarrier_u32;
    }
    static constexpr unsigned AwaitIdx0_q_produced = 0;  // Trivial size-1 ring buffer
    unsigned Parity0_q_produced = 0;
    EXO_CUDA_INLINE void Await0_q_produced(char* exo_smem, exo_ExcutThreadLog exo_excutLog, int slice) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + exo_smemOffset9_q_produced + 8*(slice * 1 + AwaitIdx0_q_produced));
      const bool enable = true;
      if (enable) {
    #if __CUDA_ARCH__ < 900
        asm volatile(
          "{\n\t"
          "// Await0_q_produced\n\t"
          ".reg.pred P1;\n\t"
          "EXO_BEFORE_WAIT:\n\t"
          "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
          "@P1 bra.uni EXO_WAIT_DONE;\n\t"
          "bra.uni EXO_BEFORE_WAIT;\n\t"
          "EXO_WAIT_DONE:\n\t"
          "}"
            :
            :"r"(mbarrier_u32),
            "r"(1u & Parity0_q_produced >> AwaitIdx0_q_produced)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_test_wait_parity_acquire_cta_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(1u & Parity0_q_produced >> AwaitIdx0_q_produced));
    #else
        asm volatile(
          "{\n\t"
          "// Await0_q_produced\n\t"
          ".reg.pred P1;\n\t"
          "EXO_BEFORE_WAIT:\n\t"
          "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
          "@P1 bra.uni EXO_WAIT_DONE;\n\t"
          "bra.uni EXO_BEFORE_WAIT;\n\t"
          "EXO_WAIT_DONE:\n\t"
          "}"
            :
            :"r"(mbarrier_u32),
            "r"(1u & Parity0_q_produced >> AwaitIdx0_q_produced)
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_try_wait_parity_acquire_cta_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(mbarrier_u32));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(1u & Parity0_q_produced >> AwaitIdx0_q_produced));
    #endif
        // Flip parity
        Parity0_q_produced ^= 1u << AwaitIdx0_q_produced;
      }
    }
#endif
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
  exo_deviceTask_consumer(
      char* exo_smem,
      exo_SyncState& exo_syncState,
      const exo_DeviceArgs& exo_deviceArgs,
      exo_Task exo_task,
      exo_ExcutThreadLog exo_excutLog={});

  static __device__ __forceinline__ void
  exo_deviceTask_producer(
      char* exo_smem,
      exo_SyncState& exo_syncState,
      const exo_DeviceArgs& exo_deviceArgs,
      exo_Task exo_task,
      exo_ExcutThreadLog exo_excutLog={});
};
}  // end inline namespace

inline void
exo_CudaInline_exocc_Sm90a_edited_tk_attn_fwd_causal::exo_Cuda0_edited_exo_tk_attn_fwd_Hdim128_causal::exo_cudaLaunch(
    cudaStream_t exo_cudaStream,
    const exo_DeviceArgs& exo_deviceArgs)
{
  namespace exo_CudaUtil = exo_CudaUtil_exocc_Sm90a_edited_tk_attn_fwd_causal;
  cudaFuncSetAttribute(exo_deviceFunction0_edited_exo_tk_attn_fwd_Hdim128_causal, cudaFuncAttributeMaxDynamicSharedMemorySize, exo_smemBytes);
#if EDIT_NO_PERSISTENT
  const unsigned exo_gridDim = unsigned(
      exo_deviceArgs.Batch * exo_deviceArgs.KV_Heads * exo_deviceArgs.Groups * ((exo_deviceArgs.SeqLen + 191u) / 192u)
  );
#else
  // TODO how expensive is it to query this every time?
  int exo_cudaDevice;
  cudaGetDevice(&exo_cudaDevice);
  int exo_SMs;
  cudaDeviceGetAttribute(&exo_SMs, cudaDevAttrMultiProcessorCount, exo_cudaDevice);
  const unsigned exo_gridDim = (unsigned(exo_SMs) & ~(exo_clusterDim - 1)) * 1u;
#endif

  cudaLaunchConfig_t exo_launchConfig = {};
  exo_launchConfig.gridDim = dim3(exo_gridDim, 1, 1);
  exo_launchConfig.blockDim = dim3(exo_blockDim, 1, 1);
  exo_launchConfig.dynamicSmemBytes = exo_smemBytes;
  exo_launchConfig.stream = exo_cudaStream;

  cudaLaunchKernelEx(&exo_launchConfig, exo_deviceFunction0_edited_exo_tk_attn_fwd_Hdim128_causal, exo_deviceArgs);

  exo_excut_flush_device_log(
      exo_cudaStream, exo_gridDim, exo_blockDim,
      exo_CudaUtil::exo_excut_str_id_count, exo_CudaUtil::exo_excut_str_table,
      1, &exo_FILE());
}

__device__ __forceinline__ void
exo_CudaInline_exocc_Sm90a_edited_tk_attn_fwd_causal::exo_Cuda0_edited_exo_tk_attn_fwd_Hdim128_causal::exo_deviceSetup(
    char* exo_smem,
    const exo_DeviceArgs& exo_deviceArgs,
    exo_ExcutThreadLog exo_excutLog)
{
  if (threadIdx.x == 0) {
    for (int i = 0; i < 2; ++i) {
        asm volatile(
          "mbarrier.init.shared::cta.b64 [%0], 384;"
            :
            :"r"(exo_smemU32(exo_smem + exo_smemOffset4_v_consumed + 8*i))
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_init_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(exo_smemU32(exo_smem + exo_smemOffset4_v_consumed + 8*i));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(384));
    }
    for (int i = 0; i < 2; ++i) {
        asm volatile(
          "mbarrier.init.shared::cta.b64 [%0], 384;"
            :
            :"r"(exo_smemU32(exo_smem + exo_smemOffset5_k_consumed + 8*i))
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_init_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(exo_smemU32(exo_smem + exo_smemOffset5_k_consumed + 8*i));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(384));
    }
    for (int i = 0; i < 1; ++i) {
        asm volatile(
          "mbarrier.init.shared::cta.b64 [%0], 384;"
            :
            :"r"(exo_smemU32(exo_smem + exo_smemOffset6_q_tmp_barrier + 8*i))
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_init_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(exo_smemU32(exo_smem + exo_smemOffset6_q_tmp_barrier + 8*i));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(384));
    }
    for (int i = 0; i < 2; ++i) {
        asm volatile(
          "mbarrier.init.shared::cta.b64 [%0], 32;"
            :
            :"r"(exo_smemU32(exo_smem + exo_smemOffset7_v_produced + 8*i))
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_init_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(exo_smemU32(exo_smem + exo_smemOffset7_v_produced + 8*i));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(32));
    }
    for (int i = 0; i < 2; ++i) {
        asm volatile(
          "mbarrier.init.shared::cta.b64 [%0], 32;"
            :
            :"r"(exo_smemU32(exo_smem + exo_smemOffset8_k_produced + 8*i))
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_init_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(exo_smemU32(exo_smem + exo_smemOffset8_k_produced + 8*i));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(32));
    }
    for (int i = 0; i < 1; ++i) {
        asm volatile(
          "mbarrier.init.shared::cta.b64 [%0], 32;"
            :
            :"r"(exo_smemU32(exo_smem + exo_smemOffset9_q_produced + 8*i))
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(mbarrier_init_shared_cta_b64), 0, __LINE__);
        exo_excutLog.log_u32_arg(exo_smemU32(exo_smem + exo_smemOffset9_q_produced + 8*i));
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(32));
    }
    asm volatile(
      "fence.proxy.async;"
    );
    exo_excutLog.log_action(EXO_EXCUT_STR_ID(fence_proxy_async), 0, __LINE__);
  }
  asm volatile(
    "barrier.cta.sync 0;"
  );
  exo_excutLog.log_action(EXO_EXCUT_STR_ID(barrier_cta_sync), 0, __LINE__);
  exo_excutLog.log_u32_arg(static_cast<uint32_t>(0));
}

__device__ __forceinline__ void
exo_CudaInline_exocc_Sm90a_edited_tk_attn_fwd_causal::exo_Cuda0_edited_exo_tk_attn_fwd_Hdim128_causal::exo_deviceTask_producer(
    char* exo_smem,
    exo_SyncState& exo_syncState,
    const exo_DeviceArgs& exo_deviceArgs,
    exo_Task exo_task,
    exo_ExcutThreadLog exo_excutLog)
{
  namespace exo_CudaUtil = exo_CudaUtil_exocc_Sm90a_edited_tk_attn_fwd_causal;
  auto& qo_smem = reinterpret_cast<exo_Sm90_SW128_tiled<exo_bf16, 2, 64, 64> (&)[]>(exo_smem[exo_smemOffset0_qo_smem]);
  auto& k_smem = reinterpret_cast<exo_Sm90_SW128_tiled<exo_bf16, 2, 128, 64> (&)[]>(exo_smem[exo_smemOffset1_k_smem]);
  auto& v_smem = reinterpret_cast<exo_Sm90_SW128_tiled<exo_bf16, 2, 128, 64> (&)[]>(exo_smem[exo_smemOffset2_v_smem]);
  ; // NO-OP
  // q_produced: barrier @ CudaMbarrier
  // k_produced: barrier @ CudaMbarrier
  // v_produced: barrier @ CudaMbarrier
  // q_tmp_barrier: barrier(q_produced) @ CudaMbarrier
  // k_consumed: barrier(k_produced) @ CudaMbarrier
  // v_consumed: barrier(v_produced) @ CudaMbarrier
  // CudaWarps(0, 1, name='producer')
  if (int CudaWarps_0_1_producer = (threadIdx.x - 384); CudaWarps_0_1_producer < 32) {
#if EDIT_MBARRIER
    kittens::semaphore& mbarrier = exo_syncState.get_q_produced(exo_smem);
#else
    // Await(q_tmp_barrier, cuda_temporal, ~1)
    exo_syncState.Await0_q_tmp_barrier(exo_smem, exo_excutLog, 0, 1);
#endif
    exo_CudaUtil::exo_Sm90_tma_to_smem(
        (&qo_smem[0])
      , exo_deviceArgs.exo_data_q_tm
      , (exo_win_2bf16_Sm90_tensorMap_128_1_1_1_64_64) { {(exo_deviceArgs.q_tm.C_offsets[0] + exo_task.batch), (exo_deviceArgs.q_tm.C_offsets[1] + exo_task.kv_head), (exo_deviceArgs.q_tm.C_offsets[2] + exo_task.group), (exo_deviceArgs.q_tm.C_offsets[3] + 192 * exo_task.qo_task), exo_deviceArgs.q_tm.C_offsets[4]} }
#if EDIT_MBARRIER
      , exo_smemU32(&mbarrier)
#else
      , exo_syncState.Arrive0_q_produced(exo_smem, exo_excutLog, 0, 0)
#endif
      , 8192
    );
    exo_CudaUtil::exo_Sm90_tma_to_smem(
        (&qo_smem[4096])
      , exo_deviceArgs.exo_data_q_tm
      , (exo_win_2bf16_Sm90_tensorMap_128_1_1_1_64_64) { {(exo_deviceArgs.q_tm.C_offsets[0] + exo_task.batch), (exo_deviceArgs.q_tm.C_offsets[1] + exo_task.kv_head), (exo_deviceArgs.q_tm.C_offsets[2] + exo_task.group), (exo_deviceArgs.q_tm.C_offsets[3] + 192 * exo_task.qo_task), (exo_deviceArgs.q_tm.C_offsets[4] + 64)} }
#if EDIT_MBARRIER
      , exo_smemU32(&mbarrier)
#else
      , exo_syncState.Arrive0_q_produced(exo_smem, exo_excutLog, 0, 0)
#endif
      , 8192
    );
    exo_CudaUtil::exo_Sm90_tma_to_smem(
        (&qo_smem[8192])
      , exo_deviceArgs.exo_data_q_tm
      , (exo_win_2bf16_Sm90_tensorMap_128_1_1_1_64_64) { {(exo_deviceArgs.q_tm.C_offsets[0] + exo_task.batch), (exo_deviceArgs.q_tm.C_offsets[1] + exo_task.kv_head), (exo_deviceArgs.q_tm.C_offsets[2] + exo_task.group), (exo_deviceArgs.q_tm.C_offsets[3] + 64 + 192 * exo_task.qo_task), exo_deviceArgs.q_tm.C_offsets[4]} }
#if EDIT_MBARRIER
      , exo_smemU32(&mbarrier)
#else
      , exo_syncState.Arrive0_q_produced(exo_smem, exo_excutLog, 0, 0)
#endif
      , 8192
    );
    exo_CudaUtil::exo_Sm90_tma_to_smem(
        (&qo_smem[12288])
      , exo_deviceArgs.exo_data_q_tm
      , (exo_win_2bf16_Sm90_tensorMap_128_1_1_1_64_64) { {(exo_deviceArgs.q_tm.C_offsets[0] + exo_task.batch), (exo_deviceArgs.q_tm.C_offsets[1] + exo_task.kv_head), (exo_deviceArgs.q_tm.C_offsets[2] + exo_task.group), (exo_deviceArgs.q_tm.C_offsets[3] + 64 + 192 * exo_task.qo_task), (exo_deviceArgs.q_tm.C_offsets[4] + 64)} }
#if EDIT_MBARRIER
      , exo_smemU32(&mbarrier)
#else
      , exo_syncState.Arrive0_q_produced(exo_smem, exo_excutLog, 0, 0)
#endif
      , 8192
    );
    exo_CudaUtil::exo_Sm90_tma_to_smem(
        (&qo_smem[16384])
      , exo_deviceArgs.exo_data_q_tm
      , (exo_win_2bf16_Sm90_tensorMap_128_1_1_1_64_64) { {(exo_deviceArgs.q_tm.C_offsets[0] + exo_task.batch), (exo_deviceArgs.q_tm.C_offsets[1] + exo_task.kv_head), (exo_deviceArgs.q_tm.C_offsets[2] + exo_task.group), (exo_deviceArgs.q_tm.C_offsets[3] + 128 + 192 * exo_task.qo_task), exo_deviceArgs.q_tm.C_offsets[4]} }
#if EDIT_MBARRIER
      , exo_smemU32(&mbarrier)
#else
      , exo_syncState.Arrive0_q_produced(exo_smem, exo_excutLog, 0, 0)
#endif
      , 8192
    );
    exo_CudaUtil::exo_Sm90_tma_to_smem(
        (&qo_smem[20480])
      , exo_deviceArgs.exo_data_q_tm
      , (exo_win_2bf16_Sm90_tensorMap_128_1_1_1_64_64) { {(exo_deviceArgs.q_tm.C_offsets[0] + exo_task.batch), (exo_deviceArgs.q_tm.C_offsets[1] + exo_task.kv_head), (exo_deviceArgs.q_tm.C_offsets[2] + exo_task.group), (exo_deviceArgs.q_tm.C_offsets[3] + 128 + 192 * exo_task.qo_task), (exo_deviceArgs.q_tm.C_offsets[4] + 64)} }
#if EDIT_MBARRIER
      , exo_smemU32(&mbarrier)
#else
      , exo_syncState.Arrive0_q_produced(exo_smem, exo_excutLog, 0, 0)
#endif
      , 8192
    );
    // Arrive(cuda_temporal, 1) >> q_produced
    // cta_mask: uint16_t(0x1)
#if EDIT_MBARRIER
    arrive(mbarrier, 1);
#else
    exo_syncState.Arrive0_q_produced(exo_smem, exo_excutLog, 0, 1);
#endif
  }
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
#if EDIT_SMART_LOOP_BOUNDS
  for (int kv_idx = 0; 128 * kv_idx < 192 + 192 * exo_task.qo_task; kv_idx++) {
    {
#else
  for (int kv_idx = 0; kv_idx < ((exo_deviceArgs.SeqLen) / (128)); kv_idx++) {
    if (128 * kv_idx < 192 + 192 * exo_task.qo_task) {
#endif
      // CudaWarps(0, 1, name='producer')
      if (int CudaWarps_0_1_producer = (threadIdx.x - 384); CudaWarps_0_1_producer < 32) {
  #if EDIT_MBARRIER
        kittens::semaphore& kc_mbarrier = exo_syncState.get_k_consumed(exo_smem)[kv_idx & 1];
        kittens::semaphore& kp_mbarrier = exo_syncState.get_k_produced(exo_smem)[kv_idx & 1];
        if (kv_idx >= 2) {
          wait(kc_mbarrier, ~((kv_idx >> 1) & 1));
        }
  #else
        // Await(k_consumed, cuda_temporal, ~2)
        exo_syncState.Await0_k_consumed(exo_smem, exo_excutLog, 0, 2);
  #endif
        exo_CudaUtil::exo_Sm90_tma_to_smem(
            (&k_smem[((kv_idx & 1) * 16384)])
          , exo_deviceArgs.exo_data_k_tm
          , (exo_win_2bf16_Sm90_tensorMap_128_1_1_128_64) { {(exo_deviceArgs.k_tm.C_offsets[0] + exo_task.batch), (exo_deviceArgs.k_tm.C_offsets[1] + exo_task.kv_head), (exo_deviceArgs.k_tm.C_offsets[2] + 128 * kv_idx), exo_deviceArgs.k_tm.C_offsets[3]} }
  #if EDIT_MBARRIER
          , exo_smemU32(&kp_mbarrier)
  #else
          , exo_syncState.Arrive0_k_produced(exo_smem, exo_excutLog, 0, 0)
  #endif
          , 16384
        );
        exo_CudaUtil::exo_Sm90_tma_to_smem(
            (&k_smem[((kv_idx & 1) * 16384 + 8192)])
          , exo_deviceArgs.exo_data_k_tm
          , (exo_win_2bf16_Sm90_tensorMap_128_1_1_128_64) { {(exo_deviceArgs.k_tm.C_offsets[0] + exo_task.batch), (exo_deviceArgs.k_tm.C_offsets[1] + exo_task.kv_head), (exo_deviceArgs.k_tm.C_offsets[2] + 128 * kv_idx), (exo_deviceArgs.k_tm.C_offsets[3] + 64)} }
  #if EDIT_MBARRIER
          , exo_smemU32(&kp_mbarrier)
  #else
          , exo_syncState.Arrive0_k_produced(exo_smem, exo_excutLog, 0, 0)
  #endif
          , 16384
        );
  #if EDIT_MBARRIER
        arrive(kp_mbarrier, 1);
  #else
        // Arrive(cuda_temporal, 1) >> k_produced
        // cta_mask: uint16_t(0x1)
        exo_syncState.Arrive0_k_produced(exo_smem, exo_excutLog, 0, 1);
  #endif

  #if EDIT_MBARRIER
        kittens::semaphore& vc_mbarrier = exo_syncState.get_v_consumed(exo_smem)[kv_idx & 1];
        kittens::semaphore& vp_mbarrier = exo_syncState.get_v_produced(exo_smem)[kv_idx & 1];
        if (kv_idx >= 2) {
          wait(vc_mbarrier, ~((kv_idx >> 1) & 1));
        }
  #else
        // Await(v_consumed, cuda_temporal, ~2)
        exo_syncState.Await0_v_consumed(exo_smem, exo_excutLog, 0, 2);
  #endif
        exo_CudaUtil::exo_Sm90_tma_to_smem(
            (&v_smem[((kv_idx & 1) * 16384)])
          , exo_deviceArgs.exo_data_v_tm
          , (exo_win_2bf16_Sm90_tensorMap_128_1_1_128_64) { {(exo_deviceArgs.v_tm.C_offsets[0] + exo_task.batch), (exo_deviceArgs.v_tm.C_offsets[1] + exo_task.kv_head), (exo_deviceArgs.v_tm.C_offsets[2] + 128 * kv_idx), exo_deviceArgs.v_tm.C_offsets[3]} }
  #if EDIT_MBARRIER
          , exo_smemU32(&vp_mbarrier)
  #else
          , exo_syncState.Arrive0_v_produced(exo_smem, exo_excutLog, 0, 0)
  #endif
          , 16384
        );
        exo_CudaUtil::exo_Sm90_tma_to_smem(
            (&v_smem[((kv_idx & 1) * 16384 + 8192)])
          , exo_deviceArgs.exo_data_v_tm
          , (exo_win_2bf16_Sm90_tensorMap_128_1_1_128_64) { {(exo_deviceArgs.v_tm.C_offsets[0] + exo_task.batch), (exo_deviceArgs.v_tm.C_offsets[1] + exo_task.kv_head), (exo_deviceArgs.v_tm.C_offsets[2] + 128 * kv_idx), (exo_deviceArgs.v_tm.C_offsets[3] + 64)} }
  #if EDIT_MBARRIER
          , exo_smemU32(&vp_mbarrier)
  #else
          , exo_syncState.Arrive0_v_produced(exo_smem, exo_excutLog, 0, 0)
  #endif
          , 16384
        );
  #if EDIT_MBARRIER
        arrive(vp_mbarrier, 1);
  #else
        // Arrive(cuda_temporal, 1) >> v_produced
        // cta_mask: uint16_t(0x1)
        exo_syncState.Arrive0_v_produced(exo_smem, exo_excutLog, 0, 1);
  #endif
      }
    }
  }
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  // Fence(cuda_in_order, cuda_generic_and_async_proxy)
  asm volatile(
    "barrier.cta.sync 0;"
  );
  exo_excutLog.log_action(EXO_EXCUT_STR_ID(barrier_cta_sync), 0, __LINE__);
  exo_excutLog.log_u32_arg(static_cast<uint32_t>(0));
  asm volatile(
    "fence.proxy.async;"
  );
  exo_excutLog.log_action(EXO_EXCUT_STR_ID(fence_proxy_async), 0, __LINE__);
  ; // NO-OP
  // Fence(cuda_in_order, cuda_in_order)
  asm volatile(
    "barrier.cta.sync 0;"
  );
  exo_excutLog.log_action(EXO_EXCUT_STR_ID(barrier_cta_sync), 0, __LINE__);
  exo_excutLog.log_u32_arg(static_cast<uint32_t>(0));
  ; // NO-OP
  // free(v_consumed)
  // free(k_consumed)
  // free(q_tmp_barrier)
  // free(v_produced)
  // free(k_produced)
  // free(q_produced)
}
__device__ __forceinline__ void
exo_CudaInline_exocc_Sm90a_edited_tk_attn_fwd_causal::exo_Cuda0_edited_exo_tk_attn_fwd_Hdim128_causal::exo_deviceTask_consumer(
    char* exo_smem,
    exo_SyncState& exo_syncState,
    const exo_DeviceArgs& exo_deviceArgs,
    exo_Task exo_task,
    exo_ExcutThreadLog exo_excutLog)
{
  namespace exo_CudaUtil = exo_CudaUtil_exocc_Sm90a_edited_tk_attn_fwd_causal;
  auto& qo_smem = reinterpret_cast<exo_Sm90_SW128_tiled<exo_bf16, 2, 64, 64> (&)[]>(exo_smem[exo_smemOffset0_qo_smem]);
  auto& k_smem = reinterpret_cast<exo_Sm90_SW128_tiled<exo_bf16, 2, 128, 64> (&)[]>(exo_smem[exo_smemOffset1_k_smem]);
  auto& v_smem = reinterpret_cast<exo_Sm90_SW128_tiled<exo_bf16, 2, 128, 64> (&)[]>(exo_smem[exo_smemOffset2_v_smem]);
  auto& lse_smem = reinterpret_cast<float (&) [192]>(exo_smem[exo_smemOffset3_lse_smem]);
  // q_produced: barrier @ CudaMbarrier
  // k_produced: barrier @ CudaMbarrier
  // v_produced: barrier @ CudaMbarrier
  // q_tmp_barrier: barrier(q_produced) @ CudaMbarrier
  // k_consumed: barrier(k_produced) @ CudaMbarrier
  // v_consumed: barrier(v_produced) @ CudaMbarrier
  ; // NO-OP
#if EDIT_MBARRIER
  {
    kittens::semaphore& mbarrier = exo_syncState.get_q_produced(exo_smem);
    wait(mbarrier, 0);
  }
#else
  // CudaWarps(name='consumer')
  if ([[maybe_unused]] int CudaWarps_None_None_consumer = threadIdx.x; 1) {
    // Await(q_produced, cuda_generic_and_async_proxy, ~0)
    exo_syncState.Await0_q_produced(exo_smem, exo_excutLog, 0);
    // Arrive(cuda_temporal, 1) >> q_tmp_barrier
    // cta_mask: uint16_t(0x1)
    exo_syncState.Arrive0_q_tmp_barrier(exo_smem, exo_excutLog, 0, 1);
  }
#endif
  exo_CudaTkScaleD<::kittens::rt_fl<16, 128, ::kittens::ducks::rt_layout::row> > att_block_d;
  exo_CudaTkScaleD<::kittens::rt_fl<16, 128, ::kittens::ducks::rt_layout::row> > att_block_scaled;
  exo_CudaTkScaleD<::kittens::rt_fl<16, 128, ::kittens::ducks::rt_layout::row> > att_block_exp2;
  exo_CudaTkScaleD<::kittens::rt_bf<16, 128, ::kittens::ducks::rt_layout::row> > att_block_a;
  exo_CudaTkScaleD<::kittens::rt_fl<16, 128, ::kittens::ducks::rt_layout::row> > o_reg;
  ::kittens::rv_fl<16, ::kittens::ducks::rv_layout::ortho> max_vec;
  ::kittens::rv_fl<16, ::kittens::ducks::rv_layout::ortho> norm_vec;
  ::kittens::rv_fl<16, ::kittens::ducks::rv_layout::ortho> max_vec_last_scaled;
  ::kittens::rv_fl<16, ::kittens::ducks::rv_layout::ortho> max_vec_last_exp2;
  ::kittens::rv_fl<16, ::kittens::ducks::rv_layout::ortho> max_vec_scaled;
  // CudaWarps(name='consumer')
  if ([[maybe_unused]] int CudaWarps_None_None_consumer = threadIdx.x; 1) {
    // cuda_threads(0, 3, unit=cuda_warpgroup)
    if ([[maybe_unused]] int exo_128thr_consumer = (threadIdx.x / 128); 1) {
      // cuda_threads(0, 4, unit=cuda_warp)
      if ([[maybe_unused]] int exo_32thr_w = (threadIdx.x % 128 / 32); 1) {
        ::kittens::warp::neg_infty(max_vec);
        ::kittens::warp::zero(norm_vec);
        ::kittens::warp::zero(o_reg.tile);
      }
    }
  }
  float scale;
  scale = 0.12751743082459868f;
#if EDIT_SMART_LOOP_BOUNDS
  for (int kv_idx = 0; 128 * kv_idx < 192 + 192 * exo_task.qo_task; kv_idx++) {
    {
#else
  for (int kv_idx = 0; kv_idx < ((exo_deviceArgs.SeqLen) / (128)); kv_idx++) {
    if (128 * kv_idx < 192 + 192 * exo_task.qo_task) {
#endif
      ; // NO-OP
      // CudaWarps(name='consumer')
      if ([[maybe_unused]] int CudaWarps_None_None_consumer = threadIdx.x; 1) {
        // cg: barrier[3] @ CudaCommitGroup
        // Await(k_produced, cuda_generic_and_async_proxy, ~0)
  #if EDIT_MBARRIER
        wait(exo_syncState.get_k_produced(exo_smem)[kv_idx & 1], (kv_idx >> 1) & 1);
  #else
        exo_syncState.Await0_k_produced(exo_smem, exo_excutLog, 0);
  #endif
        // cuda_threads(0, 3, unit=cuda_warpgroup)
        if ([[maybe_unused]] int exo_128thr_consumer = (threadIdx.x / 128); 1) {
          // Fence(wgmma_fence_1, wgmma_fence_2)
          asm volatile(
            "wgmma.fence.sync.aligned;"
          );
          exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_fence_sync_aligned), 0, __LINE__);
          att_block_d.scale_d = 0;
          #pragma unroll
          for (int hdim64 = 0; hdim64 < 2; hdim64++) {
  #if EDIT_WGMMA_DESC
            uint64_t exo_descA = exo_CudaUtil::exo_Sm90_smem_descriptor((&qo_smem[(exo_128thr_consumer * 8192 + hdim64 * 4096)]), 1, 512);
            uint64_t exo_descB = exo_CudaUtil::exo_Sm90_smem_descriptor((&k_smem[((kv_idx & 1) * 16384 + hdim64 * 8192)]), 1, 512);
            uint64_t exo_desc_stride = exo_CudaUtil::exo_Sm90_matrix_descriptor_encode(16 * sizeof(k_smem[0]));
  #endif
            {
              // K = 0 out of 64
  #if !EDIT_WGMMA_DESC
              const uint64_t exo_descA = exo_CudaUtil::exo_Sm90_smem_descriptor((&qo_smem[(exo_128thr_consumer * 8192 + hdim64 * 4096)]), 1, 512);
              const uint64_t exo_descB = exo_CudaUtil::exo_Sm90_smem_descriptor((&k_smem[((kv_idx & 1) * 16384 + hdim64 * 8192)]), 1, 512);
  #endif
              asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %64, 0;\n\t"
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %65, %66,\n\t"
                "p, 1, 1, 0, 0;\n\t"
                "}"
                  :"+f"(att_block_d.tile.tiles[0][0].data[0].x), "+f"(att_block_d.tile.tiles[0][0].data[0].y), "+f"(att_block_d.tile.tiles[0][0].data[1].x), "+f"(att_block_d.tile.tiles[0][0].data[1].y), "+f"(att_block_d.tile.tiles[0][0].data[2].x), "+f"(att_block_d.tile.tiles[0][0].data[2].y), "+f"(att_block_d.tile.tiles[0][0].data[3].x), "+f"(att_block_d.tile.tiles[0][0].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][1].data[0].x), "+f"(att_block_d.tile.tiles[0][1].data[0].y), "+f"(att_block_d.tile.tiles[0][1].data[1].x), "+f"(att_block_d.tile.tiles[0][1].data[1].y), "+f"(att_block_d.tile.tiles[0][1].data[2].x), "+f"(att_block_d.tile.tiles[0][1].data[2].y), "+f"(att_block_d.tile.tiles[0][1].data[3].x), "+f"(att_block_d.tile.tiles[0][1].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][2].data[0].x), "+f"(att_block_d.tile.tiles[0][2].data[0].y), "+f"(att_block_d.tile.tiles[0][2].data[1].x), "+f"(att_block_d.tile.tiles[0][2].data[1].y), "+f"(att_block_d.tile.tiles[0][2].data[2].x), "+f"(att_block_d.tile.tiles[0][2].data[2].y), "+f"(att_block_d.tile.tiles[0][2].data[3].x), "+f"(att_block_d.tile.tiles[0][2].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][3].data[0].x), "+f"(att_block_d.tile.tiles[0][3].data[0].y), "+f"(att_block_d.tile.tiles[0][3].data[1].x), "+f"(att_block_d.tile.tiles[0][3].data[1].y), "+f"(att_block_d.tile.tiles[0][3].data[2].x), "+f"(att_block_d.tile.tiles[0][3].data[2].y), "+f"(att_block_d.tile.tiles[0][3].data[3].x), "+f"(att_block_d.tile.tiles[0][3].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][4].data[0].x), "+f"(att_block_d.tile.tiles[0][4].data[0].y), "+f"(att_block_d.tile.tiles[0][4].data[1].x), "+f"(att_block_d.tile.tiles[0][4].data[1].y), "+f"(att_block_d.tile.tiles[0][4].data[2].x), "+f"(att_block_d.tile.tiles[0][4].data[2].y), "+f"(att_block_d.tile.tiles[0][4].data[3].x), "+f"(att_block_d.tile.tiles[0][4].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][5].data[0].x), "+f"(att_block_d.tile.tiles[0][5].data[0].y), "+f"(att_block_d.tile.tiles[0][5].data[1].x), "+f"(att_block_d.tile.tiles[0][5].data[1].y), "+f"(att_block_d.tile.tiles[0][5].data[2].x), "+f"(att_block_d.tile.tiles[0][5].data[2].y), "+f"(att_block_d.tile.tiles[0][5].data[3].x), "+f"(att_block_d.tile.tiles[0][5].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][6].data[0].x), "+f"(att_block_d.tile.tiles[0][6].data[0].y), "+f"(att_block_d.tile.tiles[0][6].data[1].x), "+f"(att_block_d.tile.tiles[0][6].data[1].y), "+f"(att_block_d.tile.tiles[0][6].data[2].x), "+f"(att_block_d.tile.tiles[0][6].data[2].y), "+f"(att_block_d.tile.tiles[0][6].data[3].x), "+f"(att_block_d.tile.tiles[0][6].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][7].data[0].x), "+f"(att_block_d.tile.tiles[0][7].data[0].y), "+f"(att_block_d.tile.tiles[0][7].data[1].x), "+f"(att_block_d.tile.tiles[0][7].data[1].y), "+f"(att_block_d.tile.tiles[0][7].data[2].x), "+f"(att_block_d.tile.tiles[0][7].data[2].y), "+f"(att_block_d.tile.tiles[0][7].data[3].x), "+f"(att_block_d.tile.tiles[0][7].data[3].y)
                  :"r"(att_block_d.scale_d),
                  "l"(exo_descA),
                  "l"(exo_descB)
              );
              exo_excutLog.log_action(EXO_EXCUT_STR_ID(setp_ne_b32), 0, __LINE__);
              exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_mma_async_sync_aligned_m64n128k16_f32_bf16_bf16), 0, __LINE__);
              att_block_d.scale_d = 1;
            }
            {
              // K = 16 out of 64
  #if !EDIT_WGMMA_DESC
              const uint64_t exo_descA = exo_CudaUtil::exo_Sm90_smem_descriptor((&qo_smem[(exo_128thr_consumer * 8192 + hdim64 * 4096 + 16)]), 1, 512);
              const uint64_t exo_descB = exo_CudaUtil::exo_Sm90_smem_descriptor((&k_smem[((kv_idx & 1) * 16384 + hdim64 * 8192 + 16)]), 1, 512);
  #else
              exo_descA += exo_desc_stride;
              exo_descB += exo_desc_stride;
  #endif
              asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %64, 0;\n\t"
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %65, %66,\n\t"
                "p, 1, 1, 0, 0;\n\t"
                "}"
                  :"+f"(att_block_d.tile.tiles[0][0].data[0].x), "+f"(att_block_d.tile.tiles[0][0].data[0].y), "+f"(att_block_d.tile.tiles[0][0].data[1].x), "+f"(att_block_d.tile.tiles[0][0].data[1].y), "+f"(att_block_d.tile.tiles[0][0].data[2].x), "+f"(att_block_d.tile.tiles[0][0].data[2].y), "+f"(att_block_d.tile.tiles[0][0].data[3].x), "+f"(att_block_d.tile.tiles[0][0].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][1].data[0].x), "+f"(att_block_d.tile.tiles[0][1].data[0].y), "+f"(att_block_d.tile.tiles[0][1].data[1].x), "+f"(att_block_d.tile.tiles[0][1].data[1].y), "+f"(att_block_d.tile.tiles[0][1].data[2].x), "+f"(att_block_d.tile.tiles[0][1].data[2].y), "+f"(att_block_d.tile.tiles[0][1].data[3].x), "+f"(att_block_d.tile.tiles[0][1].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][2].data[0].x), "+f"(att_block_d.tile.tiles[0][2].data[0].y), "+f"(att_block_d.tile.tiles[0][2].data[1].x), "+f"(att_block_d.tile.tiles[0][2].data[1].y), "+f"(att_block_d.tile.tiles[0][2].data[2].x), "+f"(att_block_d.tile.tiles[0][2].data[2].y), "+f"(att_block_d.tile.tiles[0][2].data[3].x), "+f"(att_block_d.tile.tiles[0][2].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][3].data[0].x), "+f"(att_block_d.tile.tiles[0][3].data[0].y), "+f"(att_block_d.tile.tiles[0][3].data[1].x), "+f"(att_block_d.tile.tiles[0][3].data[1].y), "+f"(att_block_d.tile.tiles[0][3].data[2].x), "+f"(att_block_d.tile.tiles[0][3].data[2].y), "+f"(att_block_d.tile.tiles[0][3].data[3].x), "+f"(att_block_d.tile.tiles[0][3].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][4].data[0].x), "+f"(att_block_d.tile.tiles[0][4].data[0].y), "+f"(att_block_d.tile.tiles[0][4].data[1].x), "+f"(att_block_d.tile.tiles[0][4].data[1].y), "+f"(att_block_d.tile.tiles[0][4].data[2].x), "+f"(att_block_d.tile.tiles[0][4].data[2].y), "+f"(att_block_d.tile.tiles[0][4].data[3].x), "+f"(att_block_d.tile.tiles[0][4].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][5].data[0].x), "+f"(att_block_d.tile.tiles[0][5].data[0].y), "+f"(att_block_d.tile.tiles[0][5].data[1].x), "+f"(att_block_d.tile.tiles[0][5].data[1].y), "+f"(att_block_d.tile.tiles[0][5].data[2].x), "+f"(att_block_d.tile.tiles[0][5].data[2].y), "+f"(att_block_d.tile.tiles[0][5].data[3].x), "+f"(att_block_d.tile.tiles[0][5].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][6].data[0].x), "+f"(att_block_d.tile.tiles[0][6].data[0].y), "+f"(att_block_d.tile.tiles[0][6].data[1].x), "+f"(att_block_d.tile.tiles[0][6].data[1].y), "+f"(att_block_d.tile.tiles[0][6].data[2].x), "+f"(att_block_d.tile.tiles[0][6].data[2].y), "+f"(att_block_d.tile.tiles[0][6].data[3].x), "+f"(att_block_d.tile.tiles[0][6].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][7].data[0].x), "+f"(att_block_d.tile.tiles[0][7].data[0].y), "+f"(att_block_d.tile.tiles[0][7].data[1].x), "+f"(att_block_d.tile.tiles[0][7].data[1].y), "+f"(att_block_d.tile.tiles[0][7].data[2].x), "+f"(att_block_d.tile.tiles[0][7].data[2].y), "+f"(att_block_d.tile.tiles[0][7].data[3].x), "+f"(att_block_d.tile.tiles[0][7].data[3].y)
                  :"r"(att_block_d.scale_d),
                  "l"(exo_descA),
                  "l"(exo_descB)
              );
              exo_excutLog.log_action(EXO_EXCUT_STR_ID(setp_ne_b32), 0, __LINE__);
              exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_mma_async_sync_aligned_m64n128k16_f32_bf16_bf16), 0, __LINE__);
              att_block_d.scale_d = 1;
            }
            {
              // K = 32 out of 64
  #if !EDIT_WGMMA_DESC
              const uint64_t exo_descA = exo_CudaUtil::exo_Sm90_smem_descriptor((&qo_smem[(exo_128thr_consumer * 8192 + hdim64 * 4096 + 32)]), 1, 512);
              const uint64_t exo_descB = exo_CudaUtil::exo_Sm90_smem_descriptor((&k_smem[((kv_idx & 1) * 16384 + hdim64 * 8192 + 32)]), 1, 512);
  #else
              exo_descA += exo_desc_stride;
              exo_descB += exo_desc_stride;
  #endif
              asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %64, 0;\n\t"
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %65, %66,\n\t"
                "p, 1, 1, 0, 0;\n\t"
                "}"
                  :"+f"(att_block_d.tile.tiles[0][0].data[0].x), "+f"(att_block_d.tile.tiles[0][0].data[0].y), "+f"(att_block_d.tile.tiles[0][0].data[1].x), "+f"(att_block_d.tile.tiles[0][0].data[1].y), "+f"(att_block_d.tile.tiles[0][0].data[2].x), "+f"(att_block_d.tile.tiles[0][0].data[2].y), "+f"(att_block_d.tile.tiles[0][0].data[3].x), "+f"(att_block_d.tile.tiles[0][0].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][1].data[0].x), "+f"(att_block_d.tile.tiles[0][1].data[0].y), "+f"(att_block_d.tile.tiles[0][1].data[1].x), "+f"(att_block_d.tile.tiles[0][1].data[1].y), "+f"(att_block_d.tile.tiles[0][1].data[2].x), "+f"(att_block_d.tile.tiles[0][1].data[2].y), "+f"(att_block_d.tile.tiles[0][1].data[3].x), "+f"(att_block_d.tile.tiles[0][1].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][2].data[0].x), "+f"(att_block_d.tile.tiles[0][2].data[0].y), "+f"(att_block_d.tile.tiles[0][2].data[1].x), "+f"(att_block_d.tile.tiles[0][2].data[1].y), "+f"(att_block_d.tile.tiles[0][2].data[2].x), "+f"(att_block_d.tile.tiles[0][2].data[2].y), "+f"(att_block_d.tile.tiles[0][2].data[3].x), "+f"(att_block_d.tile.tiles[0][2].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][3].data[0].x), "+f"(att_block_d.tile.tiles[0][3].data[0].y), "+f"(att_block_d.tile.tiles[0][3].data[1].x), "+f"(att_block_d.tile.tiles[0][3].data[1].y), "+f"(att_block_d.tile.tiles[0][3].data[2].x), "+f"(att_block_d.tile.tiles[0][3].data[2].y), "+f"(att_block_d.tile.tiles[0][3].data[3].x), "+f"(att_block_d.tile.tiles[0][3].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][4].data[0].x), "+f"(att_block_d.tile.tiles[0][4].data[0].y), "+f"(att_block_d.tile.tiles[0][4].data[1].x), "+f"(att_block_d.tile.tiles[0][4].data[1].y), "+f"(att_block_d.tile.tiles[0][4].data[2].x), "+f"(att_block_d.tile.tiles[0][4].data[2].y), "+f"(att_block_d.tile.tiles[0][4].data[3].x), "+f"(att_block_d.tile.tiles[0][4].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][5].data[0].x), "+f"(att_block_d.tile.tiles[0][5].data[0].y), "+f"(att_block_d.tile.tiles[0][5].data[1].x), "+f"(att_block_d.tile.tiles[0][5].data[1].y), "+f"(att_block_d.tile.tiles[0][5].data[2].x), "+f"(att_block_d.tile.tiles[0][5].data[2].y), "+f"(att_block_d.tile.tiles[0][5].data[3].x), "+f"(att_block_d.tile.tiles[0][5].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][6].data[0].x), "+f"(att_block_d.tile.tiles[0][6].data[0].y), "+f"(att_block_d.tile.tiles[0][6].data[1].x), "+f"(att_block_d.tile.tiles[0][6].data[1].y), "+f"(att_block_d.tile.tiles[0][6].data[2].x), "+f"(att_block_d.tile.tiles[0][6].data[2].y), "+f"(att_block_d.tile.tiles[0][6].data[3].x), "+f"(att_block_d.tile.tiles[0][6].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][7].data[0].x), "+f"(att_block_d.tile.tiles[0][7].data[0].y), "+f"(att_block_d.tile.tiles[0][7].data[1].x), "+f"(att_block_d.tile.tiles[0][7].data[1].y), "+f"(att_block_d.tile.tiles[0][7].data[2].x), "+f"(att_block_d.tile.tiles[0][7].data[2].y), "+f"(att_block_d.tile.tiles[0][7].data[3].x), "+f"(att_block_d.tile.tiles[0][7].data[3].y)
                  :"r"(att_block_d.scale_d),
                  "l"(exo_descA),
                  "l"(exo_descB)
              );
              exo_excutLog.log_action(EXO_EXCUT_STR_ID(setp_ne_b32), 0, __LINE__);
              exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_mma_async_sync_aligned_m64n128k16_f32_bf16_bf16), 0, __LINE__);
              att_block_d.scale_d = 1;
            }
            {
              // K = 48 out of 64
  #if !EDIT_WGMMA_DESC
              const uint64_t exo_descA = exo_CudaUtil::exo_Sm90_smem_descriptor((&qo_smem[(exo_128thr_consumer * 8192 + hdim64 * 4096 + 48)]), 1, 512);
              const uint64_t exo_descB = exo_CudaUtil::exo_Sm90_smem_descriptor((&k_smem[((kv_idx & 1) * 16384 + hdim64 * 8192 + 48)]), 1, 512);
  #else
              exo_descA += exo_desc_stride;
              exo_descB += exo_desc_stride;
  #endif
              asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %64, 0;\n\t"
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %65, %66,\n\t"
                "p, 1, 1, 0, 0;\n\t"
                "}"
                  :"+f"(att_block_d.tile.tiles[0][0].data[0].x), "+f"(att_block_d.tile.tiles[0][0].data[0].y), "+f"(att_block_d.tile.tiles[0][0].data[1].x), "+f"(att_block_d.tile.tiles[0][0].data[1].y), "+f"(att_block_d.tile.tiles[0][0].data[2].x), "+f"(att_block_d.tile.tiles[0][0].data[2].y), "+f"(att_block_d.tile.tiles[0][0].data[3].x), "+f"(att_block_d.tile.tiles[0][0].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][1].data[0].x), "+f"(att_block_d.tile.tiles[0][1].data[0].y), "+f"(att_block_d.tile.tiles[0][1].data[1].x), "+f"(att_block_d.tile.tiles[0][1].data[1].y), "+f"(att_block_d.tile.tiles[0][1].data[2].x), "+f"(att_block_d.tile.tiles[0][1].data[2].y), "+f"(att_block_d.tile.tiles[0][1].data[3].x), "+f"(att_block_d.tile.tiles[0][1].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][2].data[0].x), "+f"(att_block_d.tile.tiles[0][2].data[0].y), "+f"(att_block_d.tile.tiles[0][2].data[1].x), "+f"(att_block_d.tile.tiles[0][2].data[1].y), "+f"(att_block_d.tile.tiles[0][2].data[2].x), "+f"(att_block_d.tile.tiles[0][2].data[2].y), "+f"(att_block_d.tile.tiles[0][2].data[3].x), "+f"(att_block_d.tile.tiles[0][2].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][3].data[0].x), "+f"(att_block_d.tile.tiles[0][3].data[0].y), "+f"(att_block_d.tile.tiles[0][3].data[1].x), "+f"(att_block_d.tile.tiles[0][3].data[1].y), "+f"(att_block_d.tile.tiles[0][3].data[2].x), "+f"(att_block_d.tile.tiles[0][3].data[2].y), "+f"(att_block_d.tile.tiles[0][3].data[3].x), "+f"(att_block_d.tile.tiles[0][3].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][4].data[0].x), "+f"(att_block_d.tile.tiles[0][4].data[0].y), "+f"(att_block_d.tile.tiles[0][4].data[1].x), "+f"(att_block_d.tile.tiles[0][4].data[1].y), "+f"(att_block_d.tile.tiles[0][4].data[2].x), "+f"(att_block_d.tile.tiles[0][4].data[2].y), "+f"(att_block_d.tile.tiles[0][4].data[3].x), "+f"(att_block_d.tile.tiles[0][4].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][5].data[0].x), "+f"(att_block_d.tile.tiles[0][5].data[0].y), "+f"(att_block_d.tile.tiles[0][5].data[1].x), "+f"(att_block_d.tile.tiles[0][5].data[1].y), "+f"(att_block_d.tile.tiles[0][5].data[2].x), "+f"(att_block_d.tile.tiles[0][5].data[2].y), "+f"(att_block_d.tile.tiles[0][5].data[3].x), "+f"(att_block_d.tile.tiles[0][5].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][6].data[0].x), "+f"(att_block_d.tile.tiles[0][6].data[0].y), "+f"(att_block_d.tile.tiles[0][6].data[1].x), "+f"(att_block_d.tile.tiles[0][6].data[1].y), "+f"(att_block_d.tile.tiles[0][6].data[2].x), "+f"(att_block_d.tile.tiles[0][6].data[2].y), "+f"(att_block_d.tile.tiles[0][6].data[3].x), "+f"(att_block_d.tile.tiles[0][6].data[3].y),
                  "+f"(att_block_d.tile.tiles[0][7].data[0].x), "+f"(att_block_d.tile.tiles[0][7].data[0].y), "+f"(att_block_d.tile.tiles[0][7].data[1].x), "+f"(att_block_d.tile.tiles[0][7].data[1].y), "+f"(att_block_d.tile.tiles[0][7].data[2].x), "+f"(att_block_d.tile.tiles[0][7].data[2].y), "+f"(att_block_d.tile.tiles[0][7].data[3].x), "+f"(att_block_d.tile.tiles[0][7].data[3].y)
                  :"r"(att_block_d.scale_d),
                  "l"(exo_descA),
                  "l"(exo_descB)
              );
              exo_excutLog.log_action(EXO_EXCUT_STR_ID(setp_ne_b32), 0, __LINE__);
              exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_mma_async_sync_aligned_m64n128k16_f32_bf16_bf16), 0, __LINE__);
              att_block_d.scale_d = 1;
            }
          }
          // Arrive(wgmma_async, 1) >> cg[consumer]
          asm volatile(
            "wgmma.commit_group.sync.aligned;"
          );
          exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_commit_group_sync_aligned), 0, __LINE__);
          // cuda_threads(0, 4, unit=cuda_warp)
          if ([[maybe_unused]] int exo_32thr_w = (threadIdx.x % 128 / 32); 1) {
            ::kittens::warp::mul(max_vec_last_scaled, max_vec, scale);
          }
          // Await(cg[consumer], cuda_generic_and_async_proxy, 0)
          asm volatile(
            "wgmma.wait_group.sync.aligned 0;"
          );
          exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_wait_group_sync_aligned), 0, __LINE__);
          exo_excutLog.log_u32_arg(static_cast<uint32_t>(0));
        }
  #if EDIT_MBARRIER
        arrive(exo_syncState.get_k_consumed(exo_smem)[kv_idx & 1], 1);
  #else
        // Arrive(cuda_in_order, 1) >> k_consumed
        // cta_mask: uint16_t(0x1)
        exo_syncState.Arrive0_k_consumed(exo_smem, exo_excutLog, 0, 1);
  #endif
        // cuda_threads(0, 3, unit=cuda_warpgroup)
        if ([[maybe_unused]] int exo_128thr_consumer = (threadIdx.x / 128); 1) {
          // cuda_threads(0, 4, unit=cuda_warp)
          if ([[maybe_unused]] int exo_32thr_w = (threadIdx.x % 128 / 32); 1) {
            #pragma unroll
            for (int exo_causal_r = 0; exo_causal_r < 1; ++exo_causal_r) {
              #pragma unroll
              for (int exo_causal_c = 0; exo_causal_c < 8; ++exo_causal_c) {
                int exo_causal_delta = exo_causal_r * 16 - exo_causal_c * 16 + static_cast<int>((16 * exo_32thr_w + 64 * exo_128thr_consumer + 192 * exo_task.qo_task) - (128 * kv_idx));
                ::kittens::rt<float, 16, 16> exo_causal_subtile;
                exo_causal_subtile.tiles[0][0] = att_block_d.tile.tiles[exo_causal_r][exo_causal_c];
                const auto exo_causal_identity = ::kittens::base_types::constants<float>::neg_infty();
                if (exo_causal_delta < 0) ::kittens::warp::neg_infty(exo_causal_subtile);
                if (exo_causal_delta == 0) ::kittens::warp::make_causal(exo_causal_subtile, exo_causal_subtile, exo_causal_identity);
                att_block_d.tile.tiles[exo_causal_r][exo_causal_c] = exo_causal_subtile.tiles[0][0];
              }
            }
            ::kittens::warp::row_max(max_vec, att_block_d.tile, max_vec);
            ::kittens::warp::mul(att_block_scaled.tile, att_block_d.tile, scale);
            ::kittens::warp::mul(max_vec_scaled, max_vec, scale);
            ::kittens::warp::sub_row(att_block_scaled.tile, att_block_scaled.tile, max_vec_scaled);
            ::kittens::warp::exp2(att_block_exp2.tile, att_block_scaled.tile);
            ::kittens::warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            ::kittens::warp::exp2(max_vec_last_exp2, max_vec_last_scaled);
            ::kittens::warp::mul(norm_vec, norm_vec, max_vec_last_exp2);
            ::kittens::warp::row_sum(norm_vec, att_block_exp2.tile, norm_vec);
            float rmem_zero;
            rmem_zero = 0.0f;
            ::kittens::warp::add(att_block_exp2.tile, att_block_exp2.tile, rmem_zero);
            ::kittens::warp::copy(att_block_a.tile, att_block_exp2.tile);
            ::kittens::warp::mul_row(o_reg.tile, o_reg.tile, max_vec_last_exp2);
          }
        }
  #if EDIT_MBARRIER
        wait(exo_syncState.get_v_produced(exo_smem)[kv_idx & 1], (kv_idx >> 1) & 1);
  #else
        // Await(v_produced, cuda_generic_and_async_proxy, ~0)
        exo_syncState.Await0_v_produced(exo_smem, exo_excutLog, 0);
  #endif
        // cuda_threads(0, 3, unit=cuda_warpgroup)
        if ([[maybe_unused]] int exo_128thr_consumer = (threadIdx.x / 128); 1) {
          // Fence(wgmma_fence_1, wgmma_fence_2)
          asm volatile(
            "wgmma.fence.sync.aligned;"
          );
          exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_fence_sync_aligned), 0, __LINE__);
  #if EDIT_WGMMA_DESC
          uint64_t exo_descB = exo_CudaUtil::exo_Sm90_smem_descriptor((&v_smem[((kv_idx & 1) * 16384)]), 8192, 512);
          uint64_t exo_descB_stride = exo_CudaUtil::exo_Sm90_matrix_descriptor_encode(1024 * sizeof(v_smem[0]));
  #endif
          exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_fence_sync_aligned), 0, __LINE__);
          {
            // K = 0 out of 128
  #if !EDIT_WGMMA_DESC
            const uint64_t exo_descB = exo_CudaUtil::exo_Sm90_smem_descriptor((&v_smem[((kv_idx & 1) * 16384)]), 8192, 512);
  #else
  #endif
            asm volatile(
              "{\n\t"
              ".reg .pred p;\n\t"
              "setp.ne.b32 p, %64, 0;\n\t"
              "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%65, %66, %67, %68}, %69,\n\t"
              "p, 1, 1, 1;\n\t"
              "}"
                :"+f"(o_reg.tile.tiles[0][0].data[0].x), "+f"(o_reg.tile.tiles[0][0].data[0].y), "+f"(o_reg.tile.tiles[0][0].data[1].x), "+f"(o_reg.tile.tiles[0][0].data[1].y), "+f"(o_reg.tile.tiles[0][0].data[2].x), "+f"(o_reg.tile.tiles[0][0].data[2].y), "+f"(o_reg.tile.tiles[0][0].data[3].x), "+f"(o_reg.tile.tiles[0][0].data[3].y),
                "+f"(o_reg.tile.tiles[0][1].data[0].x), "+f"(o_reg.tile.tiles[0][1].data[0].y), "+f"(o_reg.tile.tiles[0][1].data[1].x), "+f"(o_reg.tile.tiles[0][1].data[1].y), "+f"(o_reg.tile.tiles[0][1].data[2].x), "+f"(o_reg.tile.tiles[0][1].data[2].y), "+f"(o_reg.tile.tiles[0][1].data[3].x), "+f"(o_reg.tile.tiles[0][1].data[3].y),
                "+f"(o_reg.tile.tiles[0][2].data[0].x), "+f"(o_reg.tile.tiles[0][2].data[0].y), "+f"(o_reg.tile.tiles[0][2].data[1].x), "+f"(o_reg.tile.tiles[0][2].data[1].y), "+f"(o_reg.tile.tiles[0][2].data[2].x), "+f"(o_reg.tile.tiles[0][2].data[2].y), "+f"(o_reg.tile.tiles[0][2].data[3].x), "+f"(o_reg.tile.tiles[0][2].data[3].y),
                "+f"(o_reg.tile.tiles[0][3].data[0].x), "+f"(o_reg.tile.tiles[0][3].data[0].y), "+f"(o_reg.tile.tiles[0][3].data[1].x), "+f"(o_reg.tile.tiles[0][3].data[1].y), "+f"(o_reg.tile.tiles[0][3].data[2].x), "+f"(o_reg.tile.tiles[0][3].data[2].y), "+f"(o_reg.tile.tiles[0][3].data[3].x), "+f"(o_reg.tile.tiles[0][3].data[3].y),
                "+f"(o_reg.tile.tiles[0][4].data[0].x), "+f"(o_reg.tile.tiles[0][4].data[0].y), "+f"(o_reg.tile.tiles[0][4].data[1].x), "+f"(o_reg.tile.tiles[0][4].data[1].y), "+f"(o_reg.tile.tiles[0][4].data[2].x), "+f"(o_reg.tile.tiles[0][4].data[2].y), "+f"(o_reg.tile.tiles[0][4].data[3].x), "+f"(o_reg.tile.tiles[0][4].data[3].y),
                "+f"(o_reg.tile.tiles[0][5].data[0].x), "+f"(o_reg.tile.tiles[0][5].data[0].y), "+f"(o_reg.tile.tiles[0][5].data[1].x), "+f"(o_reg.tile.tiles[0][5].data[1].y), "+f"(o_reg.tile.tiles[0][5].data[2].x), "+f"(o_reg.tile.tiles[0][5].data[2].y), "+f"(o_reg.tile.tiles[0][5].data[3].x), "+f"(o_reg.tile.tiles[0][5].data[3].y),
                "+f"(o_reg.tile.tiles[0][6].data[0].x), "+f"(o_reg.tile.tiles[0][6].data[0].y), "+f"(o_reg.tile.tiles[0][6].data[1].x), "+f"(o_reg.tile.tiles[0][6].data[1].y), "+f"(o_reg.tile.tiles[0][6].data[2].x), "+f"(o_reg.tile.tiles[0][6].data[2].y), "+f"(o_reg.tile.tiles[0][6].data[3].x), "+f"(o_reg.tile.tiles[0][6].data[3].y),
                "+f"(o_reg.tile.tiles[0][7].data[0].x), "+f"(o_reg.tile.tiles[0][7].data[0].y), "+f"(o_reg.tile.tiles[0][7].data[1].x), "+f"(o_reg.tile.tiles[0][7].data[1].y), "+f"(o_reg.tile.tiles[0][7].data[2].x), "+f"(o_reg.tile.tiles[0][7].data[2].y), "+f"(o_reg.tile.tiles[0][7].data[3].x), "+f"(o_reg.tile.tiles[0][7].data[3].y)
                :"r"(o_reg.scale_d),
                "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][0].data[0])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][0].data[1])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][0].data[2])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][0].data[3])),
                "l"(exo_descB)
            );
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(setp_ne_b32), 0, __LINE__);
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_mma_async_sync_aligned_m64n128k16_f32_bf16_bf16), 0, __LINE__);
            o_reg.scale_d = 1;
          }
          {
            // K = 16 out of 128
  #if !EDIT_WGMMA_DESC
            const uint64_t exo_descB = exo_CudaUtil::exo_Sm90_smem_descriptor((&v_smem[((kv_idx & 1) * 16384 + 1024)]), 8192, 512);
  #else
            exo_descB += exo_descB_stride;
  #endif
            asm volatile(
              "{\n\t"
              ".reg .pred p;\n\t"
              "setp.ne.b32 p, %64, 0;\n\t"
              "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%65, %66, %67, %68}, %69,\n\t"
              "p, 1, 1, 1;\n\t"
              "}"
                :"+f"(o_reg.tile.tiles[0][0].data[0].x), "+f"(o_reg.tile.tiles[0][0].data[0].y), "+f"(o_reg.tile.tiles[0][0].data[1].x), "+f"(o_reg.tile.tiles[0][0].data[1].y), "+f"(o_reg.tile.tiles[0][0].data[2].x), "+f"(o_reg.tile.tiles[0][0].data[2].y), "+f"(o_reg.tile.tiles[0][0].data[3].x), "+f"(o_reg.tile.tiles[0][0].data[3].y),
                "+f"(o_reg.tile.tiles[0][1].data[0].x), "+f"(o_reg.tile.tiles[0][1].data[0].y), "+f"(o_reg.tile.tiles[0][1].data[1].x), "+f"(o_reg.tile.tiles[0][1].data[1].y), "+f"(o_reg.tile.tiles[0][1].data[2].x), "+f"(o_reg.tile.tiles[0][1].data[2].y), "+f"(o_reg.tile.tiles[0][1].data[3].x), "+f"(o_reg.tile.tiles[0][1].data[3].y),
                "+f"(o_reg.tile.tiles[0][2].data[0].x), "+f"(o_reg.tile.tiles[0][2].data[0].y), "+f"(o_reg.tile.tiles[0][2].data[1].x), "+f"(o_reg.tile.tiles[0][2].data[1].y), "+f"(o_reg.tile.tiles[0][2].data[2].x), "+f"(o_reg.tile.tiles[0][2].data[2].y), "+f"(o_reg.tile.tiles[0][2].data[3].x), "+f"(o_reg.tile.tiles[0][2].data[3].y),
                "+f"(o_reg.tile.tiles[0][3].data[0].x), "+f"(o_reg.tile.tiles[0][3].data[0].y), "+f"(o_reg.tile.tiles[0][3].data[1].x), "+f"(o_reg.tile.tiles[0][3].data[1].y), "+f"(o_reg.tile.tiles[0][3].data[2].x), "+f"(o_reg.tile.tiles[0][3].data[2].y), "+f"(o_reg.tile.tiles[0][3].data[3].x), "+f"(o_reg.tile.tiles[0][3].data[3].y),
                "+f"(o_reg.tile.tiles[0][4].data[0].x), "+f"(o_reg.tile.tiles[0][4].data[0].y), "+f"(o_reg.tile.tiles[0][4].data[1].x), "+f"(o_reg.tile.tiles[0][4].data[1].y), "+f"(o_reg.tile.tiles[0][4].data[2].x), "+f"(o_reg.tile.tiles[0][4].data[2].y), "+f"(o_reg.tile.tiles[0][4].data[3].x), "+f"(o_reg.tile.tiles[0][4].data[3].y),
                "+f"(o_reg.tile.tiles[0][5].data[0].x), "+f"(o_reg.tile.tiles[0][5].data[0].y), "+f"(o_reg.tile.tiles[0][5].data[1].x), "+f"(o_reg.tile.tiles[0][5].data[1].y), "+f"(o_reg.tile.tiles[0][5].data[2].x), "+f"(o_reg.tile.tiles[0][5].data[2].y), "+f"(o_reg.tile.tiles[0][5].data[3].x), "+f"(o_reg.tile.tiles[0][5].data[3].y),
                "+f"(o_reg.tile.tiles[0][6].data[0].x), "+f"(o_reg.tile.tiles[0][6].data[0].y), "+f"(o_reg.tile.tiles[0][6].data[1].x), "+f"(o_reg.tile.tiles[0][6].data[1].y), "+f"(o_reg.tile.tiles[0][6].data[2].x), "+f"(o_reg.tile.tiles[0][6].data[2].y), "+f"(o_reg.tile.tiles[0][6].data[3].x), "+f"(o_reg.tile.tiles[0][6].data[3].y),
                "+f"(o_reg.tile.tiles[0][7].data[0].x), "+f"(o_reg.tile.tiles[0][7].data[0].y), "+f"(o_reg.tile.tiles[0][7].data[1].x), "+f"(o_reg.tile.tiles[0][7].data[1].y), "+f"(o_reg.tile.tiles[0][7].data[2].x), "+f"(o_reg.tile.tiles[0][7].data[2].y), "+f"(o_reg.tile.tiles[0][7].data[3].x), "+f"(o_reg.tile.tiles[0][7].data[3].y)
                :"r"(o_reg.scale_d),
                "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][1].data[0])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][1].data[1])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][1].data[2])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][1].data[3])),
                "l"(exo_descB)
            );
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(setp_ne_b32), 0, __LINE__);
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_mma_async_sync_aligned_m64n128k16_f32_bf16_bf16), 0, __LINE__);
            o_reg.scale_d = 1;
          }
          {
            // K = 32 out of 128
  #if !EDIT_WGMMA_DESC
            const uint64_t exo_descB = exo_CudaUtil::exo_Sm90_smem_descriptor((&v_smem[((kv_idx & 1) * 16384 + 2048)]), 8192, 512);
  #else
            exo_descB += exo_descB_stride;
  #endif
            asm volatile(
              "{\n\t"
              ".reg .pred p;\n\t"
              "setp.ne.b32 p, %64, 0;\n\t"
              "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%65, %66, %67, %68}, %69,\n\t"
              "p, 1, 1, 1;\n\t"
              "}"
                :"+f"(o_reg.tile.tiles[0][0].data[0].x), "+f"(o_reg.tile.tiles[0][0].data[0].y), "+f"(o_reg.tile.tiles[0][0].data[1].x), "+f"(o_reg.tile.tiles[0][0].data[1].y), "+f"(o_reg.tile.tiles[0][0].data[2].x), "+f"(o_reg.tile.tiles[0][0].data[2].y), "+f"(o_reg.tile.tiles[0][0].data[3].x), "+f"(o_reg.tile.tiles[0][0].data[3].y),
                "+f"(o_reg.tile.tiles[0][1].data[0].x), "+f"(o_reg.tile.tiles[0][1].data[0].y), "+f"(o_reg.tile.tiles[0][1].data[1].x), "+f"(o_reg.tile.tiles[0][1].data[1].y), "+f"(o_reg.tile.tiles[0][1].data[2].x), "+f"(o_reg.tile.tiles[0][1].data[2].y), "+f"(o_reg.tile.tiles[0][1].data[3].x), "+f"(o_reg.tile.tiles[0][1].data[3].y),
                "+f"(o_reg.tile.tiles[0][2].data[0].x), "+f"(o_reg.tile.tiles[0][2].data[0].y), "+f"(o_reg.tile.tiles[0][2].data[1].x), "+f"(o_reg.tile.tiles[0][2].data[1].y), "+f"(o_reg.tile.tiles[0][2].data[2].x), "+f"(o_reg.tile.tiles[0][2].data[2].y), "+f"(o_reg.tile.tiles[0][2].data[3].x), "+f"(o_reg.tile.tiles[0][2].data[3].y),
                "+f"(o_reg.tile.tiles[0][3].data[0].x), "+f"(o_reg.tile.tiles[0][3].data[0].y), "+f"(o_reg.tile.tiles[0][3].data[1].x), "+f"(o_reg.tile.tiles[0][3].data[1].y), "+f"(o_reg.tile.tiles[0][3].data[2].x), "+f"(o_reg.tile.tiles[0][3].data[2].y), "+f"(o_reg.tile.tiles[0][3].data[3].x), "+f"(o_reg.tile.tiles[0][3].data[3].y),
                "+f"(o_reg.tile.tiles[0][4].data[0].x), "+f"(o_reg.tile.tiles[0][4].data[0].y), "+f"(o_reg.tile.tiles[0][4].data[1].x), "+f"(o_reg.tile.tiles[0][4].data[1].y), "+f"(o_reg.tile.tiles[0][4].data[2].x), "+f"(o_reg.tile.tiles[0][4].data[2].y), "+f"(o_reg.tile.tiles[0][4].data[3].x), "+f"(o_reg.tile.tiles[0][4].data[3].y),
                "+f"(o_reg.tile.tiles[0][5].data[0].x), "+f"(o_reg.tile.tiles[0][5].data[0].y), "+f"(o_reg.tile.tiles[0][5].data[1].x), "+f"(o_reg.tile.tiles[0][5].data[1].y), "+f"(o_reg.tile.tiles[0][5].data[2].x), "+f"(o_reg.tile.tiles[0][5].data[2].y), "+f"(o_reg.tile.tiles[0][5].data[3].x), "+f"(o_reg.tile.tiles[0][5].data[3].y),
                "+f"(o_reg.tile.tiles[0][6].data[0].x), "+f"(o_reg.tile.tiles[0][6].data[0].y), "+f"(o_reg.tile.tiles[0][6].data[1].x), "+f"(o_reg.tile.tiles[0][6].data[1].y), "+f"(o_reg.tile.tiles[0][6].data[2].x), "+f"(o_reg.tile.tiles[0][6].data[2].y), "+f"(o_reg.tile.tiles[0][6].data[3].x), "+f"(o_reg.tile.tiles[0][6].data[3].y),
                "+f"(o_reg.tile.tiles[0][7].data[0].x), "+f"(o_reg.tile.tiles[0][7].data[0].y), "+f"(o_reg.tile.tiles[0][7].data[1].x), "+f"(o_reg.tile.tiles[0][7].data[1].y), "+f"(o_reg.tile.tiles[0][7].data[2].x), "+f"(o_reg.tile.tiles[0][7].data[2].y), "+f"(o_reg.tile.tiles[0][7].data[3].x), "+f"(o_reg.tile.tiles[0][7].data[3].y)
                :"r"(o_reg.scale_d),
                "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][2].data[0])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][2].data[1])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][2].data[2])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][2].data[3])),
                "l"(exo_descB)
            );
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(setp_ne_b32), 0, __LINE__);
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_mma_async_sync_aligned_m64n128k16_f32_bf16_bf16), 0, __LINE__);
            o_reg.scale_d = 1;
          }
          {
            // K = 48 out of 128
  #if !EDIT_WGMMA_DESC
            const uint64_t exo_descB = exo_CudaUtil::exo_Sm90_smem_descriptor((&v_smem[((kv_idx & 1) * 16384 + 3072)]), 8192, 512);
  #else
            exo_descB += exo_descB_stride;
  #endif
            asm volatile(
              "{\n\t"
              ".reg .pred p;\n\t"
              "setp.ne.b32 p, %64, 0;\n\t"
              "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%65, %66, %67, %68}, %69,\n\t"
              "p, 1, 1, 1;\n\t"
              "}"
                :"+f"(o_reg.tile.tiles[0][0].data[0].x), "+f"(o_reg.tile.tiles[0][0].data[0].y), "+f"(o_reg.tile.tiles[0][0].data[1].x), "+f"(o_reg.tile.tiles[0][0].data[1].y), "+f"(o_reg.tile.tiles[0][0].data[2].x), "+f"(o_reg.tile.tiles[0][0].data[2].y), "+f"(o_reg.tile.tiles[0][0].data[3].x), "+f"(o_reg.tile.tiles[0][0].data[3].y),
                "+f"(o_reg.tile.tiles[0][1].data[0].x), "+f"(o_reg.tile.tiles[0][1].data[0].y), "+f"(o_reg.tile.tiles[0][1].data[1].x), "+f"(o_reg.tile.tiles[0][1].data[1].y), "+f"(o_reg.tile.tiles[0][1].data[2].x), "+f"(o_reg.tile.tiles[0][1].data[2].y), "+f"(o_reg.tile.tiles[0][1].data[3].x), "+f"(o_reg.tile.tiles[0][1].data[3].y),
                "+f"(o_reg.tile.tiles[0][2].data[0].x), "+f"(o_reg.tile.tiles[0][2].data[0].y), "+f"(o_reg.tile.tiles[0][2].data[1].x), "+f"(o_reg.tile.tiles[0][2].data[1].y), "+f"(o_reg.tile.tiles[0][2].data[2].x), "+f"(o_reg.tile.tiles[0][2].data[2].y), "+f"(o_reg.tile.tiles[0][2].data[3].x), "+f"(o_reg.tile.tiles[0][2].data[3].y),
                "+f"(o_reg.tile.tiles[0][3].data[0].x), "+f"(o_reg.tile.tiles[0][3].data[0].y), "+f"(o_reg.tile.tiles[0][3].data[1].x), "+f"(o_reg.tile.tiles[0][3].data[1].y), "+f"(o_reg.tile.tiles[0][3].data[2].x), "+f"(o_reg.tile.tiles[0][3].data[2].y), "+f"(o_reg.tile.tiles[0][3].data[3].x), "+f"(o_reg.tile.tiles[0][3].data[3].y),
                "+f"(o_reg.tile.tiles[0][4].data[0].x), "+f"(o_reg.tile.tiles[0][4].data[0].y), "+f"(o_reg.tile.tiles[0][4].data[1].x), "+f"(o_reg.tile.tiles[0][4].data[1].y), "+f"(o_reg.tile.tiles[0][4].data[2].x), "+f"(o_reg.tile.tiles[0][4].data[2].y), "+f"(o_reg.tile.tiles[0][4].data[3].x), "+f"(o_reg.tile.tiles[0][4].data[3].y),
                "+f"(o_reg.tile.tiles[0][5].data[0].x), "+f"(o_reg.tile.tiles[0][5].data[0].y), "+f"(o_reg.tile.tiles[0][5].data[1].x), "+f"(o_reg.tile.tiles[0][5].data[1].y), "+f"(o_reg.tile.tiles[0][5].data[2].x), "+f"(o_reg.tile.tiles[0][5].data[2].y), "+f"(o_reg.tile.tiles[0][5].data[3].x), "+f"(o_reg.tile.tiles[0][5].data[3].y),
                "+f"(o_reg.tile.tiles[0][6].data[0].x), "+f"(o_reg.tile.tiles[0][6].data[0].y), "+f"(o_reg.tile.tiles[0][6].data[1].x), "+f"(o_reg.tile.tiles[0][6].data[1].y), "+f"(o_reg.tile.tiles[0][6].data[2].x), "+f"(o_reg.tile.tiles[0][6].data[2].y), "+f"(o_reg.tile.tiles[0][6].data[3].x), "+f"(o_reg.tile.tiles[0][6].data[3].y),
                "+f"(o_reg.tile.tiles[0][7].data[0].x), "+f"(o_reg.tile.tiles[0][7].data[0].y), "+f"(o_reg.tile.tiles[0][7].data[1].x), "+f"(o_reg.tile.tiles[0][7].data[1].y), "+f"(o_reg.tile.tiles[0][7].data[2].x), "+f"(o_reg.tile.tiles[0][7].data[2].y), "+f"(o_reg.tile.tiles[0][7].data[3].x), "+f"(o_reg.tile.tiles[0][7].data[3].y)
                :"r"(o_reg.scale_d),
                "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][3].data[0])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][3].data[1])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][3].data[2])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][3].data[3])),
                "l"(exo_descB)
            );
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(setp_ne_b32), 0, __LINE__);
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_mma_async_sync_aligned_m64n128k16_f32_bf16_bf16), 0, __LINE__);
            o_reg.scale_d = 1;
          }
          {
            // K = 64 out of 128
  #if !EDIT_WGMMA_DESC
            const uint64_t exo_descB = exo_CudaUtil::exo_Sm90_smem_descriptor((&v_smem[((kv_idx & 1) * 16384 + 4096)]), 8192, 512);
  #else
            exo_descB += exo_descB_stride;
  #endif
            asm volatile(
              "{\n\t"
              ".reg .pred p;\n\t"
              "setp.ne.b32 p, %64, 0;\n\t"
              "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%65, %66, %67, %68}, %69,\n\t"
              "p, 1, 1, 1;\n\t"
              "}"
                :"+f"(o_reg.tile.tiles[0][0].data[0].x), "+f"(o_reg.tile.tiles[0][0].data[0].y), "+f"(o_reg.tile.tiles[0][0].data[1].x), "+f"(o_reg.tile.tiles[0][0].data[1].y), "+f"(o_reg.tile.tiles[0][0].data[2].x), "+f"(o_reg.tile.tiles[0][0].data[2].y), "+f"(o_reg.tile.tiles[0][0].data[3].x), "+f"(o_reg.tile.tiles[0][0].data[3].y),
                "+f"(o_reg.tile.tiles[0][1].data[0].x), "+f"(o_reg.tile.tiles[0][1].data[0].y), "+f"(o_reg.tile.tiles[0][1].data[1].x), "+f"(o_reg.tile.tiles[0][1].data[1].y), "+f"(o_reg.tile.tiles[0][1].data[2].x), "+f"(o_reg.tile.tiles[0][1].data[2].y), "+f"(o_reg.tile.tiles[0][1].data[3].x), "+f"(o_reg.tile.tiles[0][1].data[3].y),
                "+f"(o_reg.tile.tiles[0][2].data[0].x), "+f"(o_reg.tile.tiles[0][2].data[0].y), "+f"(o_reg.tile.tiles[0][2].data[1].x), "+f"(o_reg.tile.tiles[0][2].data[1].y), "+f"(o_reg.tile.tiles[0][2].data[2].x), "+f"(o_reg.tile.tiles[0][2].data[2].y), "+f"(o_reg.tile.tiles[0][2].data[3].x), "+f"(o_reg.tile.tiles[0][2].data[3].y),
                "+f"(o_reg.tile.tiles[0][3].data[0].x), "+f"(o_reg.tile.tiles[0][3].data[0].y), "+f"(o_reg.tile.tiles[0][3].data[1].x), "+f"(o_reg.tile.tiles[0][3].data[1].y), "+f"(o_reg.tile.tiles[0][3].data[2].x), "+f"(o_reg.tile.tiles[0][3].data[2].y), "+f"(o_reg.tile.tiles[0][3].data[3].x), "+f"(o_reg.tile.tiles[0][3].data[3].y),
                "+f"(o_reg.tile.tiles[0][4].data[0].x), "+f"(o_reg.tile.tiles[0][4].data[0].y), "+f"(o_reg.tile.tiles[0][4].data[1].x), "+f"(o_reg.tile.tiles[0][4].data[1].y), "+f"(o_reg.tile.tiles[0][4].data[2].x), "+f"(o_reg.tile.tiles[0][4].data[2].y), "+f"(o_reg.tile.tiles[0][4].data[3].x), "+f"(o_reg.tile.tiles[0][4].data[3].y),
                "+f"(o_reg.tile.tiles[0][5].data[0].x), "+f"(o_reg.tile.tiles[0][5].data[0].y), "+f"(o_reg.tile.tiles[0][5].data[1].x), "+f"(o_reg.tile.tiles[0][5].data[1].y), "+f"(o_reg.tile.tiles[0][5].data[2].x), "+f"(o_reg.tile.tiles[0][5].data[2].y), "+f"(o_reg.tile.tiles[0][5].data[3].x), "+f"(o_reg.tile.tiles[0][5].data[3].y),
                "+f"(o_reg.tile.tiles[0][6].data[0].x), "+f"(o_reg.tile.tiles[0][6].data[0].y), "+f"(o_reg.tile.tiles[0][6].data[1].x), "+f"(o_reg.tile.tiles[0][6].data[1].y), "+f"(o_reg.tile.tiles[0][6].data[2].x), "+f"(o_reg.tile.tiles[0][6].data[2].y), "+f"(o_reg.tile.tiles[0][6].data[3].x), "+f"(o_reg.tile.tiles[0][6].data[3].y),
                "+f"(o_reg.tile.tiles[0][7].data[0].x), "+f"(o_reg.tile.tiles[0][7].data[0].y), "+f"(o_reg.tile.tiles[0][7].data[1].x), "+f"(o_reg.tile.tiles[0][7].data[1].y), "+f"(o_reg.tile.tiles[0][7].data[2].x), "+f"(o_reg.tile.tiles[0][7].data[2].y), "+f"(o_reg.tile.tiles[0][7].data[3].x), "+f"(o_reg.tile.tiles[0][7].data[3].y)
                :"r"(o_reg.scale_d),
                "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][4].data[0])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][4].data[1])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][4].data[2])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][4].data[3])),
                "l"(exo_descB)
            );
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(setp_ne_b32), 0, __LINE__);
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_mma_async_sync_aligned_m64n128k16_f32_bf16_bf16), 0, __LINE__);
            o_reg.scale_d = 1;
          }
          {
            // K = 80 out of 128
  #if !EDIT_WGMMA_DESC
            const uint64_t exo_descB = exo_CudaUtil::exo_Sm90_smem_descriptor((&v_smem[((kv_idx & 1) * 16384 + 5120)]), 8192, 512);
  #else
            exo_descB += exo_descB_stride;
  #endif
            asm volatile(
              "{\n\t"
              ".reg .pred p;\n\t"
              "setp.ne.b32 p, %64, 0;\n\t"
              "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%65, %66, %67, %68}, %69,\n\t"
              "p, 1, 1, 1;\n\t"
              "}"
                :"+f"(o_reg.tile.tiles[0][0].data[0].x), "+f"(o_reg.tile.tiles[0][0].data[0].y), "+f"(o_reg.tile.tiles[0][0].data[1].x), "+f"(o_reg.tile.tiles[0][0].data[1].y), "+f"(o_reg.tile.tiles[0][0].data[2].x), "+f"(o_reg.tile.tiles[0][0].data[2].y), "+f"(o_reg.tile.tiles[0][0].data[3].x), "+f"(o_reg.tile.tiles[0][0].data[3].y),
                "+f"(o_reg.tile.tiles[0][1].data[0].x), "+f"(o_reg.tile.tiles[0][1].data[0].y), "+f"(o_reg.tile.tiles[0][1].data[1].x), "+f"(o_reg.tile.tiles[0][1].data[1].y), "+f"(o_reg.tile.tiles[0][1].data[2].x), "+f"(o_reg.tile.tiles[0][1].data[2].y), "+f"(o_reg.tile.tiles[0][1].data[3].x), "+f"(o_reg.tile.tiles[0][1].data[3].y),
                "+f"(o_reg.tile.tiles[0][2].data[0].x), "+f"(o_reg.tile.tiles[0][2].data[0].y), "+f"(o_reg.tile.tiles[0][2].data[1].x), "+f"(o_reg.tile.tiles[0][2].data[1].y), "+f"(o_reg.tile.tiles[0][2].data[2].x), "+f"(o_reg.tile.tiles[0][2].data[2].y), "+f"(o_reg.tile.tiles[0][2].data[3].x), "+f"(o_reg.tile.tiles[0][2].data[3].y),
                "+f"(o_reg.tile.tiles[0][3].data[0].x), "+f"(o_reg.tile.tiles[0][3].data[0].y), "+f"(o_reg.tile.tiles[0][3].data[1].x), "+f"(o_reg.tile.tiles[0][3].data[1].y), "+f"(o_reg.tile.tiles[0][3].data[2].x), "+f"(o_reg.tile.tiles[0][3].data[2].y), "+f"(o_reg.tile.tiles[0][3].data[3].x), "+f"(o_reg.tile.tiles[0][3].data[3].y),
                "+f"(o_reg.tile.tiles[0][4].data[0].x), "+f"(o_reg.tile.tiles[0][4].data[0].y), "+f"(o_reg.tile.tiles[0][4].data[1].x), "+f"(o_reg.tile.tiles[0][4].data[1].y), "+f"(o_reg.tile.tiles[0][4].data[2].x), "+f"(o_reg.tile.tiles[0][4].data[2].y), "+f"(o_reg.tile.tiles[0][4].data[3].x), "+f"(o_reg.tile.tiles[0][4].data[3].y),
                "+f"(o_reg.tile.tiles[0][5].data[0].x), "+f"(o_reg.tile.tiles[0][5].data[0].y), "+f"(o_reg.tile.tiles[0][5].data[1].x), "+f"(o_reg.tile.tiles[0][5].data[1].y), "+f"(o_reg.tile.tiles[0][5].data[2].x), "+f"(o_reg.tile.tiles[0][5].data[2].y), "+f"(o_reg.tile.tiles[0][5].data[3].x), "+f"(o_reg.tile.tiles[0][5].data[3].y),
                "+f"(o_reg.tile.tiles[0][6].data[0].x), "+f"(o_reg.tile.tiles[0][6].data[0].y), "+f"(o_reg.tile.tiles[0][6].data[1].x), "+f"(o_reg.tile.tiles[0][6].data[1].y), "+f"(o_reg.tile.tiles[0][6].data[2].x), "+f"(o_reg.tile.tiles[0][6].data[2].y), "+f"(o_reg.tile.tiles[0][6].data[3].x), "+f"(o_reg.tile.tiles[0][6].data[3].y),
                "+f"(o_reg.tile.tiles[0][7].data[0].x), "+f"(o_reg.tile.tiles[0][7].data[0].y), "+f"(o_reg.tile.tiles[0][7].data[1].x), "+f"(o_reg.tile.tiles[0][7].data[1].y), "+f"(o_reg.tile.tiles[0][7].data[2].x), "+f"(o_reg.tile.tiles[0][7].data[2].y), "+f"(o_reg.tile.tiles[0][7].data[3].x), "+f"(o_reg.tile.tiles[0][7].data[3].y)
                :"r"(o_reg.scale_d),
                "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][5].data[0])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][5].data[1])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][5].data[2])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][5].data[3])),
                "l"(exo_descB)
            );
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(setp_ne_b32), 0, __LINE__);
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_mma_async_sync_aligned_m64n128k16_f32_bf16_bf16), 0, __LINE__);
            o_reg.scale_d = 1;
          }
          {
            // K = 96 out of 128
  #if !EDIT_WGMMA_DESC
            const uint64_t exo_descB = exo_CudaUtil::exo_Sm90_smem_descriptor((&v_smem[((kv_idx & 1) * 16384 + 6144)]), 8192, 512);
  #else
            exo_descB += exo_descB_stride;
  #endif
            asm volatile(
              "{\n\t"
              ".reg .pred p;\n\t"
              "setp.ne.b32 p, %64, 0;\n\t"
              "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%65, %66, %67, %68}, %69,\n\t"
              "p, 1, 1, 1;\n\t"
              "}"
                :"+f"(o_reg.tile.tiles[0][0].data[0].x), "+f"(o_reg.tile.tiles[0][0].data[0].y), "+f"(o_reg.tile.tiles[0][0].data[1].x), "+f"(o_reg.tile.tiles[0][0].data[1].y), "+f"(o_reg.tile.tiles[0][0].data[2].x), "+f"(o_reg.tile.tiles[0][0].data[2].y), "+f"(o_reg.tile.tiles[0][0].data[3].x), "+f"(o_reg.tile.tiles[0][0].data[3].y),
                "+f"(o_reg.tile.tiles[0][1].data[0].x), "+f"(o_reg.tile.tiles[0][1].data[0].y), "+f"(o_reg.tile.tiles[0][1].data[1].x), "+f"(o_reg.tile.tiles[0][1].data[1].y), "+f"(o_reg.tile.tiles[0][1].data[2].x), "+f"(o_reg.tile.tiles[0][1].data[2].y), "+f"(o_reg.tile.tiles[0][1].data[3].x), "+f"(o_reg.tile.tiles[0][1].data[3].y),
                "+f"(o_reg.tile.tiles[0][2].data[0].x), "+f"(o_reg.tile.tiles[0][2].data[0].y), "+f"(o_reg.tile.tiles[0][2].data[1].x), "+f"(o_reg.tile.tiles[0][2].data[1].y), "+f"(o_reg.tile.tiles[0][2].data[2].x), "+f"(o_reg.tile.tiles[0][2].data[2].y), "+f"(o_reg.tile.tiles[0][2].data[3].x), "+f"(o_reg.tile.tiles[0][2].data[3].y),
                "+f"(o_reg.tile.tiles[0][3].data[0].x), "+f"(o_reg.tile.tiles[0][3].data[0].y), "+f"(o_reg.tile.tiles[0][3].data[1].x), "+f"(o_reg.tile.tiles[0][3].data[1].y), "+f"(o_reg.tile.tiles[0][3].data[2].x), "+f"(o_reg.tile.tiles[0][3].data[2].y), "+f"(o_reg.tile.tiles[0][3].data[3].x), "+f"(o_reg.tile.tiles[0][3].data[3].y),
                "+f"(o_reg.tile.tiles[0][4].data[0].x), "+f"(o_reg.tile.tiles[0][4].data[0].y), "+f"(o_reg.tile.tiles[0][4].data[1].x), "+f"(o_reg.tile.tiles[0][4].data[1].y), "+f"(o_reg.tile.tiles[0][4].data[2].x), "+f"(o_reg.tile.tiles[0][4].data[2].y), "+f"(o_reg.tile.tiles[0][4].data[3].x), "+f"(o_reg.tile.tiles[0][4].data[3].y),
                "+f"(o_reg.tile.tiles[0][5].data[0].x), "+f"(o_reg.tile.tiles[0][5].data[0].y), "+f"(o_reg.tile.tiles[0][5].data[1].x), "+f"(o_reg.tile.tiles[0][5].data[1].y), "+f"(o_reg.tile.tiles[0][5].data[2].x), "+f"(o_reg.tile.tiles[0][5].data[2].y), "+f"(o_reg.tile.tiles[0][5].data[3].x), "+f"(o_reg.tile.tiles[0][5].data[3].y),
                "+f"(o_reg.tile.tiles[0][6].data[0].x), "+f"(o_reg.tile.tiles[0][6].data[0].y), "+f"(o_reg.tile.tiles[0][6].data[1].x), "+f"(o_reg.tile.tiles[0][6].data[1].y), "+f"(o_reg.tile.tiles[0][6].data[2].x), "+f"(o_reg.tile.tiles[0][6].data[2].y), "+f"(o_reg.tile.tiles[0][6].data[3].x), "+f"(o_reg.tile.tiles[0][6].data[3].y),
                "+f"(o_reg.tile.tiles[0][7].data[0].x), "+f"(o_reg.tile.tiles[0][7].data[0].y), "+f"(o_reg.tile.tiles[0][7].data[1].x), "+f"(o_reg.tile.tiles[0][7].data[1].y), "+f"(o_reg.tile.tiles[0][7].data[2].x), "+f"(o_reg.tile.tiles[0][7].data[2].y), "+f"(o_reg.tile.tiles[0][7].data[3].x), "+f"(o_reg.tile.tiles[0][7].data[3].y)
                :"r"(o_reg.scale_d),
                "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][6].data[0])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][6].data[1])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][6].data[2])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][6].data[3])),
                "l"(exo_descB)
            );
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(setp_ne_b32), 0, __LINE__);
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_mma_async_sync_aligned_m64n128k16_f32_bf16_bf16), 0, __LINE__);
            o_reg.scale_d = 1;
          }
          {
            // K = 112 out of 128
  #if !EDIT_WGMMA_DESC
            const uint64_t exo_descB = exo_CudaUtil::exo_Sm90_smem_descriptor((&v_smem[((kv_idx & 1) * 16384 + 7168)]), 8192, 512);
  #else
            exo_descB += exo_descB_stride;
  #endif
            asm volatile(
              "{\n\t"
              ".reg .pred p;\n\t"
              "setp.ne.b32 p, %64, 0;\n\t"
              "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%65, %66, %67, %68}, %69,\n\t"
              "p, 1, 1, 1;\n\t"
              "}"
                :"+f"(o_reg.tile.tiles[0][0].data[0].x), "+f"(o_reg.tile.tiles[0][0].data[0].y), "+f"(o_reg.tile.tiles[0][0].data[1].x), "+f"(o_reg.tile.tiles[0][0].data[1].y), "+f"(o_reg.tile.tiles[0][0].data[2].x), "+f"(o_reg.tile.tiles[0][0].data[2].y), "+f"(o_reg.tile.tiles[0][0].data[3].x), "+f"(o_reg.tile.tiles[0][0].data[3].y),
                "+f"(o_reg.tile.tiles[0][1].data[0].x), "+f"(o_reg.tile.tiles[0][1].data[0].y), "+f"(o_reg.tile.tiles[0][1].data[1].x), "+f"(o_reg.tile.tiles[0][1].data[1].y), "+f"(o_reg.tile.tiles[0][1].data[2].x), "+f"(o_reg.tile.tiles[0][1].data[2].y), "+f"(o_reg.tile.tiles[0][1].data[3].x), "+f"(o_reg.tile.tiles[0][1].data[3].y),
                "+f"(o_reg.tile.tiles[0][2].data[0].x), "+f"(o_reg.tile.tiles[0][2].data[0].y), "+f"(o_reg.tile.tiles[0][2].data[1].x), "+f"(o_reg.tile.tiles[0][2].data[1].y), "+f"(o_reg.tile.tiles[0][2].data[2].x), "+f"(o_reg.tile.tiles[0][2].data[2].y), "+f"(o_reg.tile.tiles[0][2].data[3].x), "+f"(o_reg.tile.tiles[0][2].data[3].y),
                "+f"(o_reg.tile.tiles[0][3].data[0].x), "+f"(o_reg.tile.tiles[0][3].data[0].y), "+f"(o_reg.tile.tiles[0][3].data[1].x), "+f"(o_reg.tile.tiles[0][3].data[1].y), "+f"(o_reg.tile.tiles[0][3].data[2].x), "+f"(o_reg.tile.tiles[0][3].data[2].y), "+f"(o_reg.tile.tiles[0][3].data[3].x), "+f"(o_reg.tile.tiles[0][3].data[3].y),
                "+f"(o_reg.tile.tiles[0][4].data[0].x), "+f"(o_reg.tile.tiles[0][4].data[0].y), "+f"(o_reg.tile.tiles[0][4].data[1].x), "+f"(o_reg.tile.tiles[0][4].data[1].y), "+f"(o_reg.tile.tiles[0][4].data[2].x), "+f"(o_reg.tile.tiles[0][4].data[2].y), "+f"(o_reg.tile.tiles[0][4].data[3].x), "+f"(o_reg.tile.tiles[0][4].data[3].y),
                "+f"(o_reg.tile.tiles[0][5].data[0].x), "+f"(o_reg.tile.tiles[0][5].data[0].y), "+f"(o_reg.tile.tiles[0][5].data[1].x), "+f"(o_reg.tile.tiles[0][5].data[1].y), "+f"(o_reg.tile.tiles[0][5].data[2].x), "+f"(o_reg.tile.tiles[0][5].data[2].y), "+f"(o_reg.tile.tiles[0][5].data[3].x), "+f"(o_reg.tile.tiles[0][5].data[3].y),
                "+f"(o_reg.tile.tiles[0][6].data[0].x), "+f"(o_reg.tile.tiles[0][6].data[0].y), "+f"(o_reg.tile.tiles[0][6].data[1].x), "+f"(o_reg.tile.tiles[0][6].data[1].y), "+f"(o_reg.tile.tiles[0][6].data[2].x), "+f"(o_reg.tile.tiles[0][6].data[2].y), "+f"(o_reg.tile.tiles[0][6].data[3].x), "+f"(o_reg.tile.tiles[0][6].data[3].y),
                "+f"(o_reg.tile.tiles[0][7].data[0].x), "+f"(o_reg.tile.tiles[0][7].data[0].y), "+f"(o_reg.tile.tiles[0][7].data[1].x), "+f"(o_reg.tile.tiles[0][7].data[1].y), "+f"(o_reg.tile.tiles[0][7].data[2].x), "+f"(o_reg.tile.tiles[0][7].data[2].y), "+f"(o_reg.tile.tiles[0][7].data[3].x), "+f"(o_reg.tile.tiles[0][7].data[3].y)
                :"r"(o_reg.scale_d),
                "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][7].data[0])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][7].data[1])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][7].data[2])), "r"(*reinterpret_cast<const uint32_t*>(&att_block_a.tile.tiles[0][7].data[3])),
                "l"(exo_descB)
            );
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(setp_ne_b32), 0, __LINE__);
            exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_mma_async_sync_aligned_m64n128k16_f32_bf16_bf16), 0, __LINE__);
            o_reg.scale_d = 1;
          }
          // Arrive(wgmma_async, 1) >> cg[consumer]
          asm volatile(
            "wgmma.commit_group.sync.aligned;"
          );
          exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_commit_group_sync_aligned), 0, __LINE__);
          // Await(cg[consumer], cuda_generic_and_async_proxy, 0)
          asm volatile(
            "wgmma.wait_group.sync.aligned 0;"
          );
          exo_excutLog.log_action(EXO_EXCUT_STR_ID(wgmma_wait_group_sync_aligned), 0, __LINE__);
          exo_excutLog.log_u32_arg(static_cast<uint32_t>(0));
        }
  #if EDIT_MBARRIER
        arrive(exo_syncState.get_v_consumed(exo_smem)[kv_idx & 1], 1);
  #else
        // free(cg)
        // Arrive(cuda_in_order, 1) >> v_consumed
        // cta_mask: uint16_t(0x1)
        exo_syncState.Arrive0_v_consumed(exo_smem, exo_excutLog, 0, 1);
  #endif
      }
    }
  }
  // CudaWarps(name='consumer')
  if ([[maybe_unused]] int CudaWarps_None_None_consumer = threadIdx.x; 1) {
    // cuda_threads(0, 3, unit=cuda_warpgroup)
    if ([[maybe_unused]] int exo_128thr_consumer = (threadIdx.x / 128); 1) {
      // cuda_threads(0, 4, unit=cuda_warp)
      if ([[maybe_unused]] int exo_32thr_w = (threadIdx.x % 128 / 32); 1) {
        ::kittens::warp::div_row(o_reg.tile, o_reg.tile, norm_vec);
        {  // Place SMEM handle in named temporary, because ThunderKittens is not const-correct.
          auto exo_tk_subtile = qo_smem[(exo_128thr_consumer * 8192)].template as_tk_subtile<2, 16, 64, ::kittens::st_bf>(0, (16 * exo_32thr_w), 0);
          ::kittens::warp::store(exo_tk_subtile, o_reg.tile);
        }
        float ln_2;
        ln_2 = 0.6931471805599453f;
        ::kittens::warp::mul(max_vec_scaled, max_vec_scaled, ln_2);
        ::kittens::rv_fl<16, ::kittens::ducks::rv_layout::ortho> norm_vec_log;
        ::kittens::warp::log(norm_vec_log, norm_vec);
        ::kittens::warp::add(norm_vec_log, norm_vec_log, max_vec_scaled);
        ::kittens::warp::store(
          exo_CudaUtil::exo_tk_cast_sv<16>((&lse_smem[exo_128thr_consumer * 64 + (16 * exo_32thr_w)])),
          norm_vec_log
        );
      }
    }
  }
  // Fence(cuda_in_order, cuda_generic_and_async_proxy)
  asm volatile(
    "barrier.cta.sync 0;"
  );
  exo_excutLog.log_action(EXO_EXCUT_STR_ID(barrier_cta_sync), 0, __LINE__);
  exo_excutLog.log_u32_arg(static_cast<uint32_t>(0));
  asm volatile(
    "fence.proxy.async;"
  );
  exo_excutLog.log_action(EXO_EXCUT_STR_ID(fence_proxy_async), 0, __LINE__);
  // CudaWarps(name='consumer')
  if ([[maybe_unused]] int CudaWarps_None_None_consumer = threadIdx.x; 1) {
    // cuda_threads(0, 3, unit=cuda_warpgroup)
    if ([[maybe_unused]] int exo_128thr_consumer = (threadIdx.x / 128); 1) {
      // CudaWarps(0, 1)
      if (int CudaWarps_0_1_consumer = (threadIdx.x % 128); CudaWarps_0_1_consumer < 32) {
        // cg: barrier @ CudaCommitGroup
        for (int hdim64 = 0; hdim64 < 2; hdim64++) {
          exo_CudaUtil::exo_Sm90_tma_to_gmem(
              exo_deviceArgs.exo_data_o_tm
            , (exo_win_2bf16_Sm90_tensorMap_128_1_1_1_64_64) { {(exo_deviceArgs.o_tm.C_offsets[0] + exo_task.batch), (exo_deviceArgs.o_tm.C_offsets[1] + exo_task.kv_head), (exo_deviceArgs.o_tm.C_offsets[2] + exo_task.group), (exo_deviceArgs.o_tm.C_offsets[3] + 64 * exo_128thr_consumer + 192 * exo_task.qo_task), (exo_deviceArgs.o_tm.C_offsets[4] + 64 * hdim64)} }
            , (&qo_smem[(exo_128thr_consumer * 8192 + hdim64 * 4096)])
          );
        }
        exo_CudaUtil::exo_Sm90_tma_to_gmem(
            exo_deviceArgs.exo_data_lse_tm
          , (exo_win_1f32_Sm90_tensorMap_0_1_1_1_64) { {(exo_deviceArgs.lse_tm.C_offsets[0] + exo_task.batch), (exo_deviceArgs.lse_tm.C_offsets[1] + exo_task.kv_head), (exo_deviceArgs.lse_tm.C_offsets[2] + exo_task.group), (exo_deviceArgs.lse_tm.C_offsets[3] + 64 * exo_128thr_consumer + 192 * exo_task.qo_task)} }
          , (&lse_smem[exo_128thr_consumer * 64])
        );
        // Arrive(tma_to_gmem_async, 1) >> cg
        asm volatile(
          "cp.async.bulk.commit_group;"
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(cp_async_bulk_commit_group), 0, __LINE__);
        // Await(cg, cuda_in_order, 0)
        asm volatile(
          "cp.async.bulk.wait_group 0;"
        );
        exo_excutLog.log_action(EXO_EXCUT_STR_ID(cp_async_bulk_wait_group), 0, __LINE__);
        exo_excutLog.log_u32_arg(static_cast<uint32_t>(0));
        // free(cg)
      }
    }
  }
  // Fence(cuda_in_order, cuda_in_order)
  asm volatile(
    "barrier.cta.sync 0;"
  );
  exo_excutLog.log_action(EXO_EXCUT_STR_ID(barrier_cta_sync), 0, __LINE__);
  exo_excutLog.log_u32_arg(static_cast<uint32_t>(0));
  // free(v_consumed)
  // free(k_consumed)
  // free(q_tmp_barrier)
  // free(v_produced)
  // free(k_produced)
  // free(q_produced)
}
__device__ __forceinline__ void
exo_CudaInline_exocc_Sm90a_edited_tk_attn_fwd_causal::exo_Cuda0_edited_exo_tk_attn_fwd_Hdim128_causal::exo_deviceMainLoop(
    char* exo_smem,
    const exo_DeviceArgs& exo_deviceArgs,
    exo_ExcutThreadLog exo_excutLog)
{
  namespace exo_CudaUtil = exo_CudaUtil_exocc_Sm90a_edited_tk_attn_fwd_causal;
  exo_SyncState exo_syncState{};
  int32_t nreg;
  nreg = ((int32_t) 0);
  if (int tmp = threadIdx.x; tmp >= 0 && tmp < 384) {
    nreg = ((int32_t) 160);
  }
  if (int tmp = threadIdx.x; tmp >= 384 && tmp < 512) {
    nreg = ((int32_t) 32);
  }
  if (nreg == ((int32_t) 32)) {
    asm("setmaxnreg.dec.sync.aligned.u32 32;");
    if (int tmp = threadIdx.x; tmp >= 384 && tmp < 512) {
      {
        int_fast32_t exo_cudaTasksLo_batch = 0;
        int_fast32_t exo_cudaTasksHi_batch = exo_deviceArgs.Batch;
        {
          int_fast32_t exo_cudaTasksLo_kv_head = 0;
          int_fast32_t exo_cudaTasksHi_kv_head = exo_deviceArgs.KV_Heads;
          {
            int_fast32_t exo_cudaTasksLo_group = 0;
            int_fast32_t exo_cudaTasksHi_group = exo_deviceArgs.Groups;
            {
              int_fast32_t exo_cudaTasksLo_qo_task = 0;
              int_fast32_t exo_cudaTasksHi_qo_task = ((191 + exo_deviceArgs.SeqLen) / (192));
              exo_TaskGenerator exo_taskGenerator(
                  blockIdx.x / exo_clusterDim,
                  gridDim.x / exo_clusterDim,
                  exo_cudaTasksLo_batch, exo_cudaTasksHi_batch,
                  exo_cudaTasksLo_kv_head, exo_cudaTasksHi_kv_head,
                  exo_cudaTasksLo_group, exo_cudaTasksHi_group,
                  exo_cudaTasksLo_qo_task, exo_cudaTasksHi_qo_task,
                  exo_deviceArgs);
#if EDIT_NO_PERSISTENT
              {
#else
              while (exo_taskGenerator.prepare_next_task()) {
#endif
                exo_deviceTask_producer(exo_smem, exo_syncState, exo_deviceArgs, exo_taskGenerator.get_next_task(), exo_excutLog);
              }
            }
          }
        }
      }
    }
  }
  if (nreg == ((int32_t) 160)) {
    asm("setmaxnreg.inc.sync.aligned.u32 160;");
    if (int tmp = threadIdx.x; tmp >= 0 && tmp < 384) {
      {
        int_fast32_t exo_cudaTasksLo_batch = 0;
        int_fast32_t exo_cudaTasksHi_batch = exo_deviceArgs.Batch;
        {
          int_fast32_t exo_cudaTasksLo_kv_head = 0;
          int_fast32_t exo_cudaTasksHi_kv_head = exo_deviceArgs.KV_Heads;
          {
            int_fast32_t exo_cudaTasksLo_group = 0;
            int_fast32_t exo_cudaTasksHi_group = exo_deviceArgs.Groups;
            {
              int_fast32_t exo_cudaTasksLo_qo_task = 0;
              int_fast32_t exo_cudaTasksHi_qo_task = ((191 + exo_deviceArgs.SeqLen) / (192));
              exo_TaskGenerator exo_taskGenerator(
                  blockIdx.x / exo_clusterDim,
                  gridDim.x / exo_clusterDim,
                  exo_cudaTasksLo_batch, exo_cudaTasksHi_batch,
                  exo_cudaTasksLo_kv_head, exo_cudaTasksHi_kv_head,
                  exo_cudaTasksLo_group, exo_cudaTasksHi_group,
                  exo_cudaTasksLo_qo_task, exo_cudaTasksHi_qo_task,
                  exo_deviceArgs);
#if EDIT_NO_PERSISTENT
              {
#else
              while (exo_taskGenerator.prepare_next_task()) {
#endif
                exo_deviceTask_consumer(exo_smem, exo_syncState, exo_deviceArgs, exo_taskGenerator.get_next_task(), exo_excutLog);
              }
            }
          }
        }
      }
    }
  }
}
