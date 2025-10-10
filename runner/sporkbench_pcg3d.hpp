#pragma once

#include <stdint.h>

namespace sporkbench {

// Copied pseudo random number generation code.
// http://www.jcgt.org/published/0009/03/02/
// Hash Functions for GPU Rendering, Mark Jarzynski, Marc Olano, NVIDIA
constexpr uint64_t pcg3d(uint32_t x, uint32_t y, uint32_t z)
{
    x = x*1664525u + 1013904223u;
    y = y*1664525u + 1013904223u;
    z = z*1664525u + 1013904223u;

    x += y*z;
    y += z*x;
    z += x*y;

    x ^= x >> 16u;
    y ^= y >> 16u;
    z ^= z >> 16u;

    x += y*z;
    y += z*x;
    z += x*y;

    return x ^ uint64_t(y) << 12u ^ uint64_t(z) << 24u;
}

}  // end namespace

