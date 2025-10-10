#pragma once

#include "sporkbench_cases.hpp"

#include <vector>

namespace sporkbench {

struct TestResult
{
    bool passed;
    double flops;
};

TestResult run_gemm_case(
        const GemmCase& gemm_case, cublasHandle_t cublasH, GemmSize size,
        float* A, float* B, float* C_test, float* C_expected,
        bool warmup, int num_trials);

}  // end namespace
