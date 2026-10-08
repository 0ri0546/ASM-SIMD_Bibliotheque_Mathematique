#pragma once

#include <cstddef>
#include <cstdint>

#include "Concepts.h"
#include "SIMD_traits.h"
#include "VecSIMD.h"
#include "MatSIMD.h"

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
// AoS - SIMD
// ============================================================

template <float_num T>
inline void DotBatchSIMDAoS(
    const Vec<T, 3>* a,
    const Vec<T, 3>* b,
    T* results,
    std::size_t count
);

template <float_num T>
inline void NormalizeBatchSIMDAoS(
    const Vec<T, 3>* input,
    Vec<T, 3>* output,
    std::size_t count
);

template <float_num T, typename Layout = RowMajor>
inline void TransformPointsBatchSIMDAoS(
    const MatSIMD<T, 4, 4, Layout>& matrix,
    const Vec<T, 3>* input,
    Vec<T, 3>* output,
    std::size_t count
);

// ============================================================
// SoA - SIMD
// ============================================================

template <float_num T>
inline void DotBatchSIMDSoA(
    const Vec3Batch<T>& a,
    const Vec3Batch<T>& b,
    T* results,
    std::size_t count
);

template <float_num T>
inline void NormalizeBatchSIMDSoA(
    const Vec3Batch<T>& input,
    const Vec3BatchOutput<T>& output,
    std::size_t count
);

template <float_num T, typename Layout = RowMajor>
inline void TransformPointsBatchSIMDSoA(
    const MatSIMD<T, 4, 4, Layout>& matrix,
    const Vec3Batch<T>& input,
    const Vec3BatchOutput<T>& output,
    std::size_t count
);

#include "BatchSIMD.inl"