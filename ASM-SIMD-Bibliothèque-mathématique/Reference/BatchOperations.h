#pragma once

#include <cstddef>

#include "Vec3Reference.h"
#include "Mat4Reference.h"

void DotBatch(
    const Vec3* a,
    const Vec3* b,
    float* output,
    std::size_t count
);

void NormalizeBatch(
    const Vec3* input,
    Vec3* output,
    std::size_t count
);

void TransformPointsBatch(
    const Vec3* input,
    Vec3* output,
    std::size_t count,
    const Mat4& matrix
);

#include "BatchOperations.inl"