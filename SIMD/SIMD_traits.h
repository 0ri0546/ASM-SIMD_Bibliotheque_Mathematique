#pragma once

#include "VecSIMD.h"

template <float_num T, std::size_t N>
struct simd_traits;

template <float_num T>
struct std::formatter<simd_traits<T, 128>>;

template <float_num T>
struct std::formatter<simd_traits<T, 256>>;


template <>
struct simd_traits<double, 128>
{
    using type = __m128d;
    static constexpr std::size_t width = 2;

    static type load(const double* data)
    {
        return _mm_loadu_pd(data);
    }
};

template <>
struct simd_traits<float, 128>
{
    using type = __m128;
    static constexpr std::size_t width = 4;

    static type load(const float* data)
    {
        return _mm_loadu_ps(data);
    }
};

template <>
struct simd_traits<float, 256>
{
    using type = __m256;
    static constexpr std::size_t width = 8;

    static type load(const float* data)
    {
        return _mm256_loadu_ps(data);
    }
};

template <>
struct simd_traits<double, 256>
{
    using type = __m256d;
    static constexpr std::size_t width = 4;

    static type load(const double* data)
    {
        return _mm256_loadu_pd(data);
    }
};