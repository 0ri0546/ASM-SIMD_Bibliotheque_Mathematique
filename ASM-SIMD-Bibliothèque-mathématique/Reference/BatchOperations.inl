#pragma once

inline void DotBatch(
    const Vec3f* a,
    const Vec3f* b,
    float* output,
    std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        output[i] = Dot(a[i], b[i]);
    }
}

inline void NormalizeBatch(
    const Vec3f* input,
    Vec3f* output,
    std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        output[i] = input[i].Normalized();
    }
}

inline void TransformPointsBatch(
    const Vec3f* input,
    Vec3f* output,
    std::size_t count,
    const Mat4f& matrix)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        output[i] = matrix.MultiplyPoint(input[i]);
    }
}
