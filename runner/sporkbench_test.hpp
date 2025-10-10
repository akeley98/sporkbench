#pragma once

#include "sporkbench_cases.hpp"

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

struct GemmTestResources
{
    cublasHandle_t cublasH;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    float* A_row_major;
    float* B_row_major;
    float* A_col_major;
    float* B_col_major;
    float* C_test;
    float* C_expected;
    size_t L2_shred_bytes;
    void* L2_shred_memory;
};

void init_test_data(const GemmTestResources& resources, GemmSize size, TestDataCode A_code, TestDataCode B_code);

TestResult run_gemm_case(
        const GemmCase& gemm_case, const GemmTestResources& resources, GemmSize size, TestCheckMode check_mode);

}  // end namespace
