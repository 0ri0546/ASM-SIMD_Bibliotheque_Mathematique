#pragma once

#include <concepts>
#include <cstdint>
#include <format>

#include "Vec.h"

template <float_num T>
class Quaternion {
public:
    using value_type = T;
    using size_type = std::size_t;

private:
    T m_x{};
    T m_y{};
    T m_z{};
    T m_w{1};

public:
    inline constexpr Quaternion() = default;

    inline constexpr Quaternion(T x, T y, T z, T w);

    template <float_num U>
    inline constexpr Quaternion(const Vec3<U>& axis, U angleRadians);

    static inline constexpr Quaternion Identity();

    template <float_num U>
    static inline constexpr Quaternion FromEuler(U pitch, U yaw, U roll);

    template <float_num U>
    static inline constexpr Quaternion FromAxisAngle(const Vec3<U>& axis, U angleRadians);

    template <float_num U>
    static inline constexpr Quaternion FromToRotation(const Vec3<U>& from, const Vec3<U>& to);

    template <float_num U>
    static inline constexpr Quaternion LookRotation(const Vec3<U>& forward, const Vec3<U>& up);

    inline constexpr T& X();
    inline constexpr const T& X() const;

    inline constexpr T& Y();
    inline constexpr const T& Y() const;

    inline constexpr T& Z();
    inline constexpr const T& Z() const;

    inline constexpr T& W();
    inline constexpr const T& W() const;

    inline constexpr T* Data();
    inline constexpr const T* Data() const;

    static inline constexpr size_type Size();

    template <float_num U>
    inline constexpr bool operator==(const Quaternion<U>& other) const;

    template <float_num U>
    inline constexpr Quaternion& operator+=(const Quaternion<U>& other);

    template <float_num U>
    inline constexpr Quaternion& operator-=(const Quaternion<U>& other);

    template <float_num U>
    inline constexpr Quaternion& operator*=(const Quaternion<U>& other);

    template <float_num U>
    inline constexpr Quaternion& operator*=(U scalar);

    template <float_num U>
    inline constexpr Quaternion& operator/=(U scalar);

    inline constexpr Quaternion operator+() const;
    inline constexpr Quaternion operator-() const;

    inline constexpr T LengthSquared() const;
    inline constexpr T Length() const;

    inline constexpr Quaternion Normalized() const;
    inline constexpr void Normalize();

    inline constexpr Quaternion Conjugate() const;
    inline constexpr Quaternion Inverse() const;

    inline constexpr Vec3<T> RotateVector(const Vec3<T>& v) const;

    inline constexpr Vec3<T> ToEuler() const;

    inline constexpr T Pitch() const;
    inline constexpr T Yaw() const;
    inline constexpr T Roll() const;

    inline constexpr void ToAxisAngle(Vec3<T>& outAxis, T& outAngleRadians) const;
};

template <float_num T, float_num U>
using QuaternionCommon = Quaternion<std::common_type_t<T, U>>;

template <float_num T, float_num U>
inline constexpr auto operator+(const Quaternion<T>& a, const Quaternion<U>& b) -> QuaternionCommon<T, U>;

template <float_num T, float_num U>
inline constexpr auto operator-(const Quaternion<T>& a, const Quaternion<U>& b) -> QuaternionCommon<T, U>;

template <float_num T, float_num U>
inline constexpr auto operator*(const Quaternion<T>& a, const Quaternion<U>& b) -> QuaternionCommon<T, U>;

template <float_num T, float_num U>
inline constexpr auto operator*(const Quaternion<T>& q, U scalar) -> QuaternionCommon<T, U>;

template <float_num T, float_num U>
inline constexpr auto operator*(U scalar, const Quaternion<T>& q) -> QuaternionCommon<T, U>;

template <float_num T, float_num U>
inline constexpr auto operator/(const Quaternion<T>& q, U scalar) -> QuaternionCommon<T, U>;

template <float_num T, float_num U>
inline constexpr auto operator*(const Quaternion<T>& q, const Vec3<U>& v) -> Vec3<std::common_type_t<T, U>>;

template <float_num T>
inline constexpr T Dot(const Quaternion<T>& a, const Quaternion<T>& b);

template <float_num T>
inline constexpr Quaternion<T> Lerp(const Quaternion<T>& a, const Quaternion<T>& b, T t);

template <float_num T>
inline constexpr Quaternion<T> Slerp(const Quaternion<T>& a, const Quaternion<T>& b, T t);

template <float_num T>
inline constexpr Quaternion<T> LerpUnclamped(const Quaternion<T>& a, const Quaternion<T>& b, T t);

template <float_num T>
inline constexpr Quaternion<T> SlerpUnclamped(const Quaternion<T>& a, const Quaternion<T>& b, T t);

template <float_num T>
inline constexpr T Angle(const Quaternion<T>& a, const Quaternion<T>& b);

template <float_num T>
inline constexpr Quaternion<T> RotateTowards(const Quaternion<T>& from, const Quaternion<T>& to, T maxAngleRadians);

using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

template <float_num T>
struct std::formatter<Quaternion<T>>;

#include "Quaternion.inl"
