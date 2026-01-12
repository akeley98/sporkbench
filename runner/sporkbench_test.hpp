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
    random = 0,
    batch_index_identity = 1,
    tiled_numbers = 2,
    signs_only = 3,
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
    Ctype* C_expected;
    size_t L2_shred_bytes;
    void* L2_shred_memory;
};

using GemmTestResourcesUnion = std::variant<
    GemmTestResourcesT<float, float>,
    GemmTestResourcesT<float, __half>,
    GemmTestResourcesT<__half, __half>
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

void init_test_data(GemmTestResourcesUnion resources, GemmSize size, TestDataCode A_code, TestDataCode B_code);

TestResult run_gemm_case(
        GemmCaseUnion gemm_case, GemmTestResourcesUnion resources, GemmSize size, TestCheckMode check_mode);

void init_test_data(const GemvTestResources& resources, GemvSize size, TestDataCode A_code, TestDataCode B_code);

TestResult run_gemv_case(
        const GemvCase& gemv_case, const GemvTestResources& resources, GemvSize size, TestCheckMode check_mode);

}  // end namespace
