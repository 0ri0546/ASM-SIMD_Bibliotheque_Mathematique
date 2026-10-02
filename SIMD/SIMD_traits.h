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

    static void store(double* data, type vec)
    {
        return _mm_storeu_pd(data, vec);
    }

    static double* from (type data)
    {
        return reinterpret_cast<double*>(&data);
    }

    static type set1(double scalar)
    {
        return _mm_set1_pd(scalar);
    }

    static type add(type a, type b)
    {
        return _mm_add_pd(a, b);
    }

    static type sub(type a, type b)
    {
        return _mm_sub_pd(a, b);
    }

    static type mul(type a, type b)
    {
        return _mm_mul_pd(a, b);
    }

    static type div(type a, type b)
    {
        return _mm_div_pd(a, b);
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

    static void store(float* data, type vec)
    {
        return _mm_storeu_ps(data, vec);
    }

    static float* from(type data)
    {
        return reinterpret_cast<float*>(&data);
    }

    static type set1(float scalar)
    {
        return _mm_set_ps1(scalar);
    }

    static type add(type a, type b)
    {
        return _mm_add_ps(a, b);
    }

    static type sub(type a, type b)
    {
        return _mm_sub_ps(a, b);
    }

    static type mul(type a, type b)
    {
        return _mm_mul_ps(a, b);
    }

    static type div(type a, type b)
    {
        return _mm_div_ps(a, b);
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

    static void store(float* data, type vec)
    {
        return _mm256_storeu_ps(data, vec);
    }

    static float* from(type data)
    {
        return reinterpret_cast<float*>(&data);
    }

    static type set1(float scalar)
    {
        return _mm256_set1_ps(scalar);
    }

    static type add(type a, type b)
    {
        return _mm256_add_ps(a, b);
    }

    static type sub(type a, type b)
    {
        return _mm256_sub_ps(a, b);
    }

    static type mul(type a, type b)
    {
        return _mm256_mul_ps(a, b);
    }

    static type div(type a, type b)
    {
        return _mm256_div_ps(a, b);
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

    static void store(double* data, type vec)
    {
        return _mm256_storeu_pd(data, vec);
    }

    static double* from(type data)
    {
        return reinterpret_cast<double*>(&data);
    }

    static type set1(double scalar)
    {
        return _mm256_set1_pd(scalar);
    }

    static type add(type a, type b)
    {
        return _mm256_add_pd(a, b);
    }

    static type sub(type a, type b)
    {
        return _mm256_sub_pd(a, b);
    }

    static type mul(type a, type b)
    {
        return _mm256_mul_pd(a, b);
    }

    static type div(type a, type b)
    {
        return _mm256_div_pd(a, b);
    }
};
