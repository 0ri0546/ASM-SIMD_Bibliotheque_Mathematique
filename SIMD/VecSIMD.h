#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <format>
#include <cmath>
#include <type_traits>
#include <utility>

#include <immintrin.h>

#include "Concepts.h"
#include "SIMD_traits.h"

template <float_num T, std::size_t N>
class VecSIMD
{
public:
    using value_type = T;
    using size_type = std::size_t;

private:
    std::array<T, N> m_data{};

public:
    inline constexpr VecSIMD() = default;

    template <typename... Args>
        requires(sizeof...(Args) == N) &&
    (std::convertible_to<Args, T> && ...)
        inline constexpr VecSIMD(Args&&... args);

    inline constexpr T& operator[](size_type index);
    inline constexpr const T& operator[](size_type index) const;

    template <std::size_t I>
    inline auto get_m128() const;

    template <std::size_t I>
    inline auto get_m256() const;

    inline constexpr T* Data();
    inline constexpr const T* Data() const;

    static inline constexpr size_type Size();

    template <float_num U>
    inline constexpr bool operator==(const VecSIMD<U, N>& other) const;

    template <float_num U>
    inline constexpr VecSIMD& operator+=(const VecSIMD<U, N>& other);

    template <float_num U>
    inline constexpr VecSIMD& operator-=(const VecSIMD<U, N>& other);

    template <float_num U>
    inline constexpr VecSIMD& operator*=(const VecSIMD<U, N>& other);

    template <float_num U>
    inline constexpr VecSIMD& operator/=(const VecSIMD<U, N>& other);

    template <float_num U>
    inline constexpr VecSIMD& operator*=(U scalar);

    template <float_num U>
    inline constexpr VecSIMD& operator/=(U scalar);

    inline constexpr VecSIMD operator+() const;
    inline constexpr VecSIMD operator-() const;

    inline constexpr T LengthSquared() const;
    inline constexpr T Length() const;

    inline constexpr VecSIMD Normalized() const;
    inline constexpr void Normalize();
};

template <float_num T, float_num U, std::size_t N>
using VecSIMDCommon = VecSIMD<std::common_type_t<T, U>, N>;

template <float_num T, float_num U, std::size_t N>
inline auto operator+(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
    ) -> VecSIMDCommon<T, U, N>;

template <float_num T, float_num U, std::size_t N>
inline auto operator-(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
    ) -> VecSIMDCommon<T, U, N>;

template <float_num T, float_num U, std::size_t N>
inline auto operator*(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
    ) -> VecSIMDCommon<T, U, N>;

template <float_num T, float_num U, std::size_t N>
inline auto operator/(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
    ) -> VecSIMDCommon<T, U, N>;

template <float_num T, float_num U, std::size_t N>
inline auto operator*(
    const VecSIMD<T, N>& v,
    U scalar
    ) -> VecSIMDCommon<T, U, N>;

template <float_num T, float_num U, std::size_t N>
inline auto operator*(
    U scalar,
    const VecSIMD<T, N>& v
    ) -> VecSIMDCommon<T, U, N>;

template <float_num T, float_num U, std::size_t N>
inline auto operator/(
    const VecSIMD<T, N>& v,
    U scalar
    ) -> VecSIMDCommon<T, U, N>;

template <float_num T, float_num U, std::size_t N>
inline auto Dot(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
) -> std::common_type_t<T, U>;

template <>
struct std::formatter<__m128>;

template <>
struct std::formatter<__m128d>;

template <>
struct std::formatter<__m256>;

template <>
struct std::formatter<__m256d>;

template <float_num T, std::size_t N>
struct std::formatter<VecSIMD<T, N>>;

#include "VecSIMD.inl"
