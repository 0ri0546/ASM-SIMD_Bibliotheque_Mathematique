#pragma once

#include <cstdint>

// ============================================================
// AoS - SIMD
// ============================================================

template <float_num T>
inline void DotBatchSIMDAoS(
    const Vec<T, 3>* a,
    const Vec<T, 3>* b,
    T* results,
    std::size_t count
)
{
    Detail::ForEachBatch<T>(count, [&]<typename traits>(std::size_t i)
    {
        std::int32_t indices[traits::width];

        for (std::size_t k = 0; k < traits::width; ++k)
            indices[k] = static_cast<std::int32_t>(k * 3);

        const T* aData = reinterpret_cast<const T*>(a + i);
        const T* bData = reinterpret_cast<const T*>(b + i);

        const auto ax = traits::gather(aData, indices);
        const auto ay = traits::gather(aData + 1, indices);
        const auto az = traits::gather(aData + 2, indices);

        const auto bx = traits::gather(bData, indices);
        const auto by = traits::gather(bData + 1, indices);
        const auto bz = traits::gather(bData + 2, indices);

        const auto result = traits::add(
            traits::add(
                traits::mul(ax, bx),
                traits::mul(ay, by)
            ),
            traits::mul(az, bz)
        );

        traits::store(results + i, result);
    });
}

template <float_num T>
inline void NormalizeBatchSIMDAoS(
    const Vec<T, 3>* input,
    Vec<T, 3>* output,
    std::size_t count
)
{
    Detail::ForEachBatch<T>(count, [&]<typename traits>(std::size_t i)
    {
        std::int32_t indices[traits::width];

        for (std::size_t k = 0; k < traits::width; ++k)
            indices[k] = static_cast<std::int32_t>(k * 3);

        const T* inputData = reinterpret_cast<const T*>(input + i);

        const auto x = traits::gather(inputData, indices);
        const auto y = traits::gather(inputData + 1, indices);
        const auto z = traits::gather(inputData + 2, indices);

        const auto lengthSquared = traits::add(
            traits::add(
                traits::mul(x, x),
                traits::mul(y, y)
            ),
            traits::mul(z, z)
        );

        const auto zero = traits::set1(T{ 0 });
        const auto one = traits::set1(T{ 1 });

        const auto zeroMask =
            traits::compareEqual(lengthSquared, zero);

        const auto safeLengthSquared =
            traits::select(
                lengthSquared,
                one,
                zeroMask
            );

        const auto length =
            traits::sqrt(safeLengthSquared);

        const auto inverseLength =
            traits::div(one, length);

        const auto resultX =
            traits::mul(x, inverseLength);

        const auto resultY =
            traits::mul(y, inverseLength);

        const auto resultZ =
            traits::mul(z, inverseLength);

        T valuesX[traits::width];
        T valuesY[traits::width];
        T valuesZ[traits::width];

        traits::store(valuesX, resultX);
        traits::store(valuesY, resultY);
        traits::store(valuesZ, resultZ);

        for (std::size_t k = 0; k < traits::width; ++k)
        {
            output[i + k][0] = valuesX[k];
            output[i + k][1] = valuesY[k];
            output[i + k][2] = valuesZ[k];
        }
    });
}

template <float_num T, typename Layout>
inline void TransformPointsBatchSIMDAoS(
    const MatSIMD<T, 4, 4, Layout>& matrix,
    const Vec<T, 3>* input,
    Vec<T, 3>* output,
    std::size_t count
)
{
    Detail::ForEachBatch<T>(count, [&]<typename traits>(std::size_t i)
    {
        std::int32_t indices[traits::width];

        for (std::size_t k = 0; k < traits::width; ++k)
            indices[k] = static_cast<std::int32_t>(k * 3);

        const T* inputData = reinterpret_cast<const T*>(input + i);

        const auto x = traits::gather(inputData, indices);
        const auto y = traits::gather(inputData + 1, indices);
        const auto z = traits::gather(inputData + 2, indices);

        const auto m00 = traits::set1(matrix[0, 0]);
        const auto m01 = traits::set1(matrix[0, 1]);
        const auto m02 = traits::set1(matrix[0, 2]);
        const auto m03 = traits::set1(matrix[0, 3]);

        const auto m10 = traits::set1(matrix[1, 0]);
        const auto m11 = traits::set1(matrix[1, 1]);
        const auto m12 = traits::set1(matrix[1, 2]);
        const auto m13 = traits::set1(matrix[1, 3]);

        const auto m20 = traits::set1(matrix[2, 0]);
        const auto m21 = traits::set1(matrix[2, 1]);
        const auto m22 = traits::set1(matrix[2, 2]);
        const auto m23 = traits::set1(matrix[2, 3]);

        const auto resultX = traits::add(
            traits::add(
                traits::mul(m00, x),
                traits::mul(m01, y)
            ),
            traits::add(
                traits::mul(m02, z),
                m03
            )
        );

        const auto resultY = traits::add(
            traits::add(
                traits::mul(m10, x),
                traits::mul(m11, y)
            ),
            traits::add(
                traits::mul(m12, z),
                m13
            )
        );

        const auto resultZ = traits::add(
            traits::add(
                traits::mul(m20, x),
                traits::mul(m21, y)
            ),
            traits::add(
                traits::mul(m22, z),
                m23
            )
        );

        T valuesX[traits::width];
        T valuesY[traits::width];
        T valuesZ[traits::width];

        traits::store(valuesX, resultX);
        traits::store(valuesY, resultY);
        traits::store(valuesZ, resultZ);

        for (std::size_t k = 0; k < traits::width; ++k)
        {
            output[i + k][0] = valuesX[k];
            output[i + k][1] = valuesY[k];
            output[i + k][2] = valuesZ[k];
        }
    });
}

// ============================================================
// SoA - SIMD
// ============================================================

template <float_num T>
inline void DotBatchSIMDSoA(
    const Vec3Batch<T>& a,
    const Vec3Batch<T>& b,
    T* results,
    std::size_t count
)
{
    Detail::ForEachBatch<T>(count, [&]<typename traits>(std::size_t i)
    {
        const auto ax = traits::load(a.x + i);
        const auto ay = traits::load(a.y + i);
        const auto az = traits::load(a.z + i);

        const auto bx = traits::load(b.x + i);
        const auto by = traits::load(b.y + i);
        const auto bz = traits::load(b.z + i);

        const auto result = traits::add(
            traits::add(
                traits::mul(ax, bx),
                traits::mul(ay, by)
            ),
            traits::mul(az, bz)
        );

        traits::store(results + i, result);
    });
}

template <float_num T>
inline void NormalizeBatchSIMDSoA(
    const Vec3Batch<T>& input,
    const Vec3BatchOutput<T>& output,
    std::size_t count
)
{
    Detail::ForEachBatch<T>(count, [&]<typename traits>(std::size_t i)
    {
        const auto x = traits::load(input.x + i);
        const auto y = traits::load(input.y + i);
        const auto z = traits::load(input.z + i);

        const auto lengthSquared = traits::add(
            traits::add(
                traits::mul(x, x),
                traits::mul(y, y)
            ),
            traits::mul(z, z)
        );

        const auto zero = traits::set1(T{ 0 });
        const auto one = traits::set1(T{ 1 });

        const auto zeroMask =
            traits::compareEqual(lengthSquared, zero);

        const auto safeLengthSquared =
            traits::select(
                lengthSquared,
                one,
                zeroMask
            );

        const auto length =
            traits::sqrt(safeLengthSquared);

        const auto inverseLength =
            traits::div(one, length);

        traits::store(
            output.x + i,
            traits::mul(x, inverseLength)
        );

        traits::store(
            output.y + i,
            traits::mul(y, inverseLength)
        );

        traits::store(
            output.z + i,
            traits::mul(z, inverseLength)
        );
    });
}

template <float_num T, typename Layout>
inline void TransformPointsBatchSIMDSoA(
    const MatSIMD<T, 4, 4, Layout>& matrix,
    const Vec3Batch<T>& input,
    const Vec3BatchOutput<T>& output,
    std::size_t count
)
{
    Detail::ForEachBatch<T>(count, [&]<typename traits>(std::size_t i)
    {
        const auto x = traits::load(input.x + i);
        const auto y = traits::load(input.y + i);
        const auto z = traits::load(input.z + i);

        const auto m00 = traits::set1(matrix[0, 0]);
        const auto m01 = traits::set1(matrix[0, 1]);
        const auto m02 = traits::set1(matrix[0, 2]);
        const auto m03 = traits::set1(matrix[0, 3]);

        const auto m10 = traits::set1(matrix[1, 0]);
        const auto m11 = traits::set1(matrix[1, 1]);
        const auto m12 = traits::set1(matrix[1, 2]);
        const auto m13 = traits::set1(matrix[1, 3]);

        const auto m20 = traits::set1(matrix[2, 0]);
        const auto m21 = traits::set1(matrix[2, 1]);
        const auto m22 = traits::set1(matrix[2, 2]);
        const auto m23 = traits::set1(matrix[2, 3]);

        const auto rx = traits::add(
            traits::add(
                traits::mul(m00, x),
                traits::mul(m01, y)
            ),
            traits::add(
                traits::mul(m02, z),
                m03
            )
        );

        const auto ry = traits::add(
            traits::add(
                traits::mul(m10, x),
                traits::mul(m11, y)
            ),
            traits::add(
                traits::mul(m12, z),
                m13
            )
        );

        const auto rz = traits::add(
            traits::add(
                traits::mul(m20, x),
                traits::mul(m21, y)
            ),
            traits::add(
                traits::mul(m22, z),
                m23
            )
        );

        traits::store(output.x + i, rx);
        traits::store(output.y + i, ry);
        traits::store(output.z + i, rz);
    });
}