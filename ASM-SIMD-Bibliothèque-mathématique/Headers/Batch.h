#pragma once

#include <cstddef>
#include <cstdint>

#include "../Concepts.h"
#include "Vec.h"
#include "Mat.h"

#ifndef VEC3_BATCH_DEFINED
#define VEC3_BATCH_DEFINED

template <float_num T>
struct Vec3Batch
{
    const T* x;
    const T* y;
    const T* z;
};

template <float_num T>
struct Vec3BatchOutput
{
    T* x;
    T* y;
    T* z;
};

#endif

// ============================================================
// AoS - Reference
// ============================================================

template <float_num T>
inline void DotBatchAoS(
    const Vec<T, 3>* a,
    const Vec<T, 3>* b,
    T* results,
    std::size_t count
);

template <float_num T>
inline void NormalizeBatchAoS(
    const Vec<T, 3>* input,
    Vec<T, 3>* output,
    std::size_t count
);

template <float_num T, typename Layout = RowMajor>
inline void TransformPointsBatchAoS(
    const Mat<T, 4, 4, Layout>& matrix,
    const Vec<T, 3>* input,
    Vec<T, 3>* output,
    std::size_t count
);

// ============================================================
// SoA - Reference
// ============================================================

template <float_num T>
inline void DotBatchSoA(
    const Vec3Batch<T>& a,
    const Vec3Batch<T>& b,
    T* results,
    std::size_t count
);

template <float_num T>
inline void NormalizeBatchSoA(
    const Vec3Batch<T>& input,
    const Vec3BatchOutput<T>& output,
    std::size_t count
);

template <float_num T, typename Layout = RowMajor>
inline void TransformPointsBatchSoA(
    const Mat<T, 4, 4, Layout>& matrix,
    const Vec3Batch<T>& input,
    const Vec3BatchOutput<T>& output,
    std::size_t count
);

#include "Batch.inl"