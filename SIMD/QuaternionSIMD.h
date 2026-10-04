#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <format>
#include <cmath>
#include <numbers>
#include <type_traits>
#include <utility>

#include <immintrin.h>

#include "Concepts.h"
#include "VecSIMD.h"
#include "SIMD_traits.h"


template <float_num T>
class QuaternionSIMD;

template <float_num T, float_num U>
using QuaternionCommonSIMD = QuaternionSIMD<std::common_type_t<T, U>>;

template <float_num T>
class QuaternionSIMD
{
public:

    constexpr QuaternionSIMD() = default;

    constexpr QuaternionSIMD(T x, T y, T z, T w);

    template <float_num U>
    constexpr QuaternionSIMD(
        const VecSIMD<U, 3>& axis,
        U angleRadians
    );

    static constexpr QuaternionSIMD Identity();

    template <float_num U>
    static constexpr QuaternionSIMD FromEuler(
        U pitch,
        U yaw,
        U roll
    );

    template <float_num U>
    static constexpr QuaternionSIMD FromAxisAngle(
        const VecSIMD<U, 3>& axis,
        U angleRadians
    );

    template <float_num U>
    static constexpr QuaternionSIMD FromToRotation(
        const VecSIMD<U, 3>& from,
        const VecSIMD<U, 3>& to
    );

    template <float_num U>
    static constexpr QuaternionSIMD LookRotation(
        const VecSIMD<U, 3>& forward,
        const VecSIMD<U, 3>& up
    );

    constexpr T& X();
    constexpr const T& X() const;

    constexpr T& Y();
    constexpr const T& Y() const;

    constexpr T& Z();
    constexpr const T& Z() const;

    constexpr T& W();
    constexpr const T& W() const;

    constexpr T* Data();
    constexpr const T* Data() const;

    static constexpr std::size_t Size();

    template <float_num U>
    constexpr bool operator==(const QuaternionSIMD<U>& other) const;

    template <float_num U>
    constexpr QuaternionSIMD& operator+=(
        const QuaternionSIMD<U>& other
        );

    template <float_num U>
    constexpr QuaternionSIMD& operator-=(
        const QuaternionSIMD<U>& other
        );

    template <float_num U>
    constexpr QuaternionSIMD& operator*=(
        const QuaternionSIMD<U>& other
        );

    template <float_num U>
    constexpr QuaternionSIMD& operator*=(U scalar);

    template <float_num U>
    constexpr QuaternionSIMD& operator/=(U scalar);

    constexpr QuaternionSIMD operator+() const;
    constexpr QuaternionSIMD operator-() const;

    constexpr T LengthSquared() const;
    constexpr T Length() const;

    constexpr QuaternionSIMD Normalized() const;
    constexpr void Normalize();

    constexpr QuaternionSIMD Conjugate() const;
    constexpr QuaternionSIMD Inverse() const;

    constexpr VecSIMD<T, 3> RotateVector(
        const VecSIMD<T, 3>& v
    ) const;

    constexpr VecSIMD<T, 3> ToEuler() const;

    constexpr T Pitch() const;
    constexpr T Yaw() const;
    constexpr T Roll() const;

    constexpr void ToAxisAngle(
        VecSIMD<T, 3>& outAxis,
        T& outAngleRadians
    ) const;


private:

    VecSIMD<T, 4> m_data{
        T{ 0 },
        T{ 0 },
        T{ 0 },
        T{ 1 }
    };
};

template <float_num T, float_num U>
auto constexpr operator+(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<U>& b
    ) -> QuaternionCommonSIMD<T, U>;


template <float_num T, float_num U>
auto constexpr operator-(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<U>& b
    ) -> QuaternionCommonSIMD<T, U>;


template <float_num T, float_num U>
auto constexpr operator*(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<U>& b
    ) -> QuaternionCommonSIMD<T, U>;


template <float_num T, float_num U>
auto constexpr operator*(
    const QuaternionSIMD<T>& q,
    U scalar
    ) -> QuaternionCommonSIMD<T, U>;


template <float_num T, float_num U>
auto constexpr operator*(
    U scalar,
    const QuaternionSIMD<T>& q
    ) -> QuaternionCommonSIMD<T, U>;


template <float_num T, float_num U>
auto constexpr operator/(
    const QuaternionSIMD<T>& q,
    U scalar
    ) -> QuaternionCommonSIMD<T, U>;


template <float_num T, float_num U>
auto constexpr operator*(
    const QuaternionSIMD<T>& q,
    const VecSIMD<U, 3>& v
    ) -> VecSIMD<std::common_type_t<T, U>, 3>;

template <float_num T>
constexpr T Dot(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<T>& b
);


template <float_num T>
constexpr QuaternionSIMD<T> LerpUnclamped(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<T>& b,
    T t
);


template <float_num T>
constexpr QuaternionSIMD<T> Lerp(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<T>& b,
    T t
);


template <float_num T>
constexpr QuaternionSIMD<T> SlerpUnclamped(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<T>& b,
    T t
);


template <float_num T>
constexpr QuaternionSIMD<T> Slerp(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<T>& b,
    T t
);


template <float_num T>
constexpr T Angle(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<T>& b
);


template <float_num T>
constexpr QuaternionSIMD<T> RotateTowards(
    const QuaternionSIMD<T>& from,
    const QuaternionSIMD<T>& to,
    T maxAngleRadians
);

template <float_num T>
struct std::formatter<QuaternionSIMD<T>>;

using QuaternionSIMDf = QuaternionSIMD<float>;
using QuaternionSIMDd = QuaternionSIMD<double>;

#include "QuaternionSIMD.inl"