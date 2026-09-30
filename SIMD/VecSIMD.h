#pragma once

#include <array>
#include <concepts>
#include <cstdint>
#include <format>

#include <xmmintrin.h>

template <std::floating_point T, std::size_t N>
class Vec {
public:
    using value_type = T;
    using size_type = std::size_t;

private:
    std::array<T, N> m_data{};

public:
    inline constexpr Vec() = default;

    template <typename... Args>
        requires(sizeof...(Args) == N) && (std::convertible_to<Args, T> && ...)
    inline constexpr Vec(Args&&... args);

    inline constexpr T& operator[](size_type index);
    inline constexpr const T& operator[](size_type index) const;

	template <std::size_t I>
    inline auto get_m128() const;
    template <std::size_t I>
    inline auto get_m256() const;

    inline constexpr T* Data();
    inline constexpr const T* Data() const;

    static inline constexpr size_type Size();

    template <std::floating_point U>
    inline constexpr bool operator==(const Vec<U, N>& other) const;

    template <std::floating_point U>
    inline constexpr Vec& operator+=(const Vec<U, N>& other);

    template <std::floating_point U>
    inline constexpr Vec& operator-=(const Vec<U, N>& other);

    template <std::floating_point U>
    inline constexpr Vec& operator*=(const Vec<U, N>& other);

    template <std::floating_point U>
    inline constexpr Vec& operator/=(const Vec<U, N>& other);

    template <std::floating_point U>
    inline constexpr Vec& operator*=(U scalar);

    template <std::floating_point U>
    inline constexpr Vec& operator/=(U scalar);

    inline constexpr Vec operator+() const;
    inline constexpr Vec operator-() const;

    inline constexpr T LengthSquared() const;
    inline constexpr T Length() const;

    inline constexpr Vec Normalized() const;
    inline constexpr void Normalize();
};

template <std::floating_point T, std::floating_point U, std::size_t N>
using VecCommon = Vec<std::common_type_t<T, U>, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto operator+(const Vec<T, N>& a, const Vec<U, N>& b) -> VecCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto operator-(const Vec<T, N>& a, const Vec<U, N>& b) -> VecCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto operator*(const Vec<T, N>& a, const Vec<U, N>& b) -> VecCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto operator/(const Vec<T, N>& a, const Vec<U, N>& b) -> VecCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto operator*(const Vec<T, N>& v, U scalar) -> VecCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto operator*(U scalar, const Vec<T, N>& v) -> VecCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto operator/(const Vec<T, N>& v, U scalar) -> VecCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto Dot(const Vec<T, N>& a, const Vec<U, N>& b) -> std::common_type_t<T, U>;

template <std::floating_point T, std::floating_point U>
inline constexpr auto Cross(const Vec<T, 3>& a, const Vec<U, 3>& b) -> VecCommon<T, U, 3>;

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto Distance(const Vec<T, N>& a, const Vec<U, N>& b) -> std::common_type_t<T, U>;

template <std::floating_point T, std::floating_point U, std::floating_point V, std::size_t N>
inline constexpr auto Lerp(const Vec<T, N>& a, const Vec<U, N>& b, V t) -> VecCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto Scale(const Vec<T, N>& a, const Vec<U, N>& b) -> VecCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto Min(const Vec<T, N>& a, const Vec<U, N>& b) -> VecCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto Max(const Vec<T, N>& a, const Vec<U, N>& b) -> VecCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::floating_point V, std::size_t N>
inline constexpr auto MoveTowards(const Vec<T, N>& current, const Vec<U, N>& target, V maxDistanceDelta)
-> VecCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto Reflect(const Vec<T, N>& v, const Vec<U, N>& normal) -> VecCommon<T, U, N>;

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto Angle(const Vec<T, N>& a, const Vec<U, N>& b) -> std::common_type_t<T, U>;

template <std::floating_point T>
inline constexpr Vec<T, 2> Perpendicular(const Vec<T, 2>& v);

template <std::floating_point T>
using Vec2 = Vec<T, 2>;

template <std::floating_point T>
using Vec3 = Vec<T, 3>;

template <std::floating_point T>
using Vec4 = Vec<T, 4>;

using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec4f = Vec4<float>;

using Vec2d = Vec2<double>;
using Vec3d = Vec3<double>;
using Vec4d = Vec4<double>;

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
struct std::formatter<Vec<T, N>>;

#include "VecSIMD.inl"
