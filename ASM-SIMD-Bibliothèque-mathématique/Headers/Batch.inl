#pragma once

#include <cmath>

// ============================================================
// AoS - Reference
// ============================================================

template <float_num T>
inline void DotBatchAoS(
    const Vec<T, 3>* a,
    const Vec<T, 3>* b,
    T* results,
    std::size_t count
)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        results[i] = Dot(a[i], b[i]);
    }
}

template <float_num T>
inline void NormalizeBatchAoS(
    const Vec<T, 3>* input,
    Vec<T, 3>* output,
    std::size_t count
)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        output[i] = input[i].Normalized();
    }
}

template <float_num T, typename Layout>
inline void TransformPointsBatchAoS(
    const Mat<T, 4, 4, Layout>& matrix,
    const Vec<T, 3>* input,
    Vec<T, 3>* output,
    std::size_t count
)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        output[i] = matrix.MultiplyPoint(input[i]);
    }
}

// ============================================================
// SoA - Reference
// ============================================================

template <float_num T>
inline void DotBatchSoA(
    const Vec3Batch<T>& a,
    const Vec3Batch<T>& b,
    T* results,
    std::size_t count
)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        results[i] =
            a.x[i] * b.x[i] +
            a.y[i] * b.y[i] +
            a.z[i] * b.z[i];
    }
}

template <float_num T>
inline void NormalizeBatchSoA(
    const Vec3Batch<T>& input,
    const Vec3BatchOutput<T>& output,
    std::size_t count
)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        const T lengthSquared =
            input.x[i] * input.x[i] +
            input.y[i] * input.y[i] +
            input.z[i] * input.z[i];

        if (lengthSquared == T{ 0 })
        {
            output.x[i] = T{ 0 };
            output.y[i] = T{ 0 };
            output.z[i] = T{ 0 };
            continue;
        }

        const T inverseLength =
            T{ 1 } / std::sqrt(lengthSquared);

        output.x[i] = input.x[i] * inverseLength;
        output.y[i] = input.y[i] * inverseLength;
        output.z[i] = input.z[i] * inverseLength;
    }
}

template <float_num T, typename Layout>
inline void TransformPointsBatchSoA(
    const Mat<T, 4, 4, Layout>& matrix,
    const Vec3Batch<T>& input,
    const Vec3BatchOutput<T>& output,
    std::size_t count
)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        const T x = input.x[i];
        const T y = input.y[i];
        const T z = input.z[i];

        output.x[i] =
            matrix[0, 0] * x +
            matrix[0, 1] * y +
            matrix[0, 2] * z +
            matrix[0, 3];

        output.y[i] =
            matrix[1, 0] * x +
            matrix[1, 1] * y +
            matrix[1, 2] * z +
            matrix[1, 3];

        output.z[i] =
            matrix[2, 0] * x +
            matrix[2, 1] * y +
            matrix[2, 2] * z +
            matrix[2, 3];
    }
}