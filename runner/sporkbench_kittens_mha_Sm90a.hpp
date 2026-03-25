#pragma once

#include "sporkbench_cases.hpp"

namespace sporkbench {
namespace kittens_mha_h100 {

// scale_lse: if true, multiplies the lse by -sqrt(Hdim).
// This is expected by the thunder kittens backwards pass, but is non-standard.
// Also, they have an undocumented mandatory 192 divisor (non-power-of-2!!!!) for seq_len.
void attention_forward(
        int batch, int kv_heads, int Groups, int seq_len, int head_dim, bool is_causal, bool scale_lse,
        exo_bf16* o_ptr, float* l_ptr, exo_bf16* q_ptr, exo_bf16* k_ptr, exo_bf16* v_ptr,
        cudaStream_t stream);

void attention_backward(
        int batch, int kv_heads, int Groups, int seq_len, int head_dim, bool is_causal,
        exo_bf16* qg_ptr, exo_bf16* kg_ptr, exo_bf16* vg_ptr, float* d_ptr,
        exo_bf16* q_ptr, exo_bf16* k_ptr, exo_bf16* v_ptr, exo_bf16* o_ptr, float* l_ptr, exo_bf16* og_ptr,
        cudaStream_t stream);

}  // end namespace
}  // end namespace
