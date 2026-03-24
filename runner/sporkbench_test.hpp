#pragma once

#include "sporkbench_cases.hpp"

#include <variant>
#include <vector>

namespace sporkbench {

struct TestResult
{
    bool passed;
    double flops;
};

enum class TestCheckMode
{
    none = 0,
    approximate = 1,
    exact = 2,
};

enum class TestDataCode
{
    random_no_outliers = 0,
    random_with_outliers = 1,
    batch_index_identity = 2,
    tiled_numbers = 3,
    signs_only = 4,
};

template <typename Ctype, typename ABtype>
struct GemmTestResourcesT
{
    cublasHandle_t cublasH;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    ABtype* A_row_major;
    ABtype* B_row_major;
    ABtype* A_col_major;
    ABtype* B_col_major;
    Ctype* C_test;
    Ctype* C_expected_row_major;
    Ctype* C_expected_col_major;
    size_t L2_shred_bytes;
    void* L2_shred_memory;
};

static_assert(std::variant_size_v<GemmCaseUnion> == 6, "Update GemmTestResourcesUnion");

using GemmTestResourcesUnion = std::variant<
    GemmTestResourcesT<float, float>,
    GemmTestResourcesT<float, __half>,
    GemmTestResourcesT<__half, __half>,
    GemmTestResourcesT<float, __nv_bfloat16>,
    GemmTestResourcesT<float, exo_e4m3>,
    GemmTestResourcesT<float, exo_e5m2>
>;

struct GemvTestResources
{
    cublasHandle_t cublasH;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    float* A;
    float* x;
    float* y_test;
    float* y_expected;
    size_t L2_shred_bytes;
    void* L2_shred_memory;
};

template <typename T_type, typename L_type, int Hdim, bool Causal>
struct AttnFwdTestResourcesT
{
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    T_type* d_O_test;
    L_type* d_l_vec_test;
    T_type* d_O_expected;
    L_type* d_l_vec_expected;
    T_type* d_Q;
    T_type* d_K;
    T_type* d_V;
    size_t L2_shred_bytes;
    void* L2_shred_memory;
};

static_assert(std::variant_size_v<AttnFwdCaseUnion> == 4, "Update AttnFwdCaseUnion");

using AttnFwdTestResourcesUnion = std::variant<
    AttnFwdTestResourcesT<exo_bf16, float, 64, false>,
    AttnFwdTestResourcesT<exo_bf16, float, 64, true>,
    AttnFwdTestResourcesT<exo_bf16, float, 128, false>,
    AttnFwdTestResourcesT<exo_bf16, float, 128, true>
>;

void init_test_data(GemmTestResourcesUnion resources, GemmSize size, TestDataCode A_code, TestDataCode B_code);

TestResult run_gemm_case(
        GemmCaseUnion gemm_case, GemmTestResourcesUnion resources, GemmSize size, TestCheckMode check_mode);

void init_test_data(const GemvTestResources& resources, GemvSize size, TestDataCode A_code, TestDataCode B_code);

TestResult run_gemv_case(
        const GemvCase& gemv_case, const GemvTestResources& resources, GemvSize size, TestCheckMode check_mode);

void init_test_data(
        const AttnFwdTestResourcesUnion& resources, AttnFwdSize size,
        TestDataCode Q_code, TestDataCode K_code, TestDataCode V_code);

TestResult run_attn_fwd_case(
        AttnFwdCaseUnion attn_fwd_case, AttnFwdTestResourcesUnion resources, AttnFwdSize size, TestCheckMode check_mode);



}  // end namespace
