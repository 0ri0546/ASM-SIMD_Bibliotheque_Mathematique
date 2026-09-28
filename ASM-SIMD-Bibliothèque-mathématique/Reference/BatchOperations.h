#pragma once

#include <cstddef>

#include "../Headers/Vec.h"
#include "../Headers/Mat.h"

void DotBatch(
    const Vec3f* a,
    const Vec3f* b,
    float* output,
    std::size_t count
);

void NormalizeBatch(
    const Vec3f* input,
    Vec3f* output,
    std::size_t count
);

void TransformPointsBatch(
    const Vec3f* input,
    Vec3f* output,
    std::size_t count,
    const Mat4f& matrix
);

#include "BatchOperations.inl"
