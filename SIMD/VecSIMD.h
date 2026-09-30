#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <format>
#include <cmath>
#include <type_traits>
#include <utility>
#include <immintrin.h>

template <std::floating_point T, std::size_t N>
class VecSIMD
{
public:
    using value_type = T;
    using size_type = std::size_t;

private:
    std::array<T, N> m_data{};

public:
    constexpr VecSIMD() = default;

    template <typename... Args>
        requires(sizeof...(Args) == N) &&
    (std::convertible_to<Args, T> && ...)
        constexpr VecSIMD(Args&&... args);

    constexpr T& operator[](size_type index);
    constexpr const T& operator[](size_type index) const;

    template <std::size_t I>
    auto get_m128() const;

    template <std::size_t I>
    auto get_m256() const;

    constexpr T* Data();
    constexpr const T* Data() const;

    static constexpr size_type Size();

    template <std::floating_point U>
    constexpr bool operator==(const VecSIMD<U, N>& other) const;

    template <std::floating_point U>
    constexpr VecSIMD& operator+=(const VecSIMD<U, N>& other);

    template <std::floating_point U>
    constexpr VecSIMD& operator-=(const VecSIMD<U, N>& other);

    template <std::floating_point U>
    constexpr VecSIMD& operator*=(const VecSIMD<U, N>& other);

    template <std::floating_point U>
    constexpr VecSIMD& operator/=(const VecSIMD<U, N>& other);

    template <std::floating_point U>
    constexpr VecSIMD& operator*=(U scalar);

    template <std::floating_point U>
    constexpr VecSIMD& operator/=(U scalar);

    constexpr VecSIMD operator+() const;
    constexpr VecSIMD operator-() const;

    constexpr T LengthSquared() const;
    constexpr T Length() const;

    constexpr VecSIMD Normalized() const;
    constexpr void Normalize();
};

template <std::floating_point T, std::floating_point U, std::size_t N>
using VecSIMDCommon = VecSIMD<std::common_type_t<T, U>, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
auto operator+(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
    ) -> VecSIMDCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
auto operator-(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
    ) -> VecSIMDCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
auto operator*(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
    ) -> VecSIMDCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
auto operator/(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
    ) -> VecSIMDCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
auto operator*(
    const VecSIMD<T, N>& v,
    U scalar
    ) -> VecSIMDCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
auto operator*(
    U scalar,
    const VecSIMD<T, N>& v
    ) -> VecSIMDCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
auto operator/(
    const VecSIMD<T, N>& v,
    U scalar
    ) -> VecSIMDCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
auto Dot(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
) -> std::common_type_t<T, U>;

template <std::floating_point T, std::size_t N>
struct simd_traits;

template <std::floating_point T>
struct std::formatter<simd_traits<T, 128>>;

template <std::floating_point T>
struct std::formatter<simd_traits<T, 256>>;

template <>
struct std::formatter<__m128>;

template <>
struct std::formatter<__m128d>;

template <>
struct std::formatter<__m256>;

template <>
struct std::formatter<__m256d>;

template <std::floating_point T, std::size_t N>
struct std::formatter<VecSIMD<T, N>>;

#include "VecSIMD.inl"