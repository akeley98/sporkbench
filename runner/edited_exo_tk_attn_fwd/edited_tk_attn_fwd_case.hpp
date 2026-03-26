#pragma once

#include "exocc_Sm90a_edited_tk_attn_fwd.h"

#include "sporkbench_cases.hpp"

namespace sporkbench {

inline const AttnFwdCase_bf16_f32_128 edited_tk_attn_fwd_case_bf16_f32_128
{
    CudaArch::Sm90a,
    "builtin",
    "edited_tk_attn_fwd_bf16_f32_128",
    [] (AttnFwdSize size, exo_bf16* O, float* lse, const exo_bf16* Q, const exo_bf16* K, const exo_bf16* V) {
        edited_exo_tk_attn_fwd_Hdim128(nullptr, size.Batch, size.KV_Heads, size.Groups, size.SeqLen, O, lse, Q, K, V);
    },
    1, INT32_MAX,
    1, INT32_MAX,
    1, INT32_MAX,
    64, INT32_MAX,
    false
};

}
