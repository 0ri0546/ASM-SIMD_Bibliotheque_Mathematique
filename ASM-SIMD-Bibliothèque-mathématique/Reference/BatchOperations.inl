#pragma once

inline void DotBatch(
    const Vec3* a,
    const Vec3* b,
    float* output,
    std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        output[i] = Vec3::Dot(a[i], b[i]);
    }
}

inline void NormalizeBatch(
    const Vec3* input,
    Vec3* output,
    std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        output[i] = input[i].normalized();
    }
}

inline void TransformPointsBatch(
    const Vec3* input,
    Vec3* output,
    std::size_t count,
    const Mat4& matrix)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        output[i] = matrix.MultiplyPoint3x4(input[i]);
    }
}