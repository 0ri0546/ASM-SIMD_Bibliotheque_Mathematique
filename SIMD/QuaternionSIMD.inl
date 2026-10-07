#pragma once

namespace Detail {
    template <float_num T>
    inline T HorizontalSumSquared(const T* data)
    {
        if constexpr (std::same_as<T, float>)
        {
            const __m128 v = _mm_loadu_ps(data);
            const __m128 mul = _mm_mul_ps(v, v);
            const __m128 h1 = _mm_hadd_ps(mul, mul);
            const __m128 h2 = _mm_hadd_ps(h1, h1);

            return _mm_cvtss_f32(h2);
        }
        else
        {
            const __m256d v = _mm256_loadu_pd(data);
            const __m256d mul = _mm256_mul_pd(v, v);
            const __m256d h = _mm256_hadd_pd(mul, mul);

            const __m128d low = _mm256_castpd256_pd128(h);
            const __m128d high = _mm256_extractf128_pd(h, 1);
            const __m128d sum = _mm_add_pd(low, high);

            return _mm_cvtsd_f64(sum);
        }
    }

    template <float_num T>
    inline auto LoadVector3(const VecSIMD<T, 3>& v)
    {
        if constexpr (std::same_as<T, float>)
        {
            return _mm_set_ps(
                0.0f,
                v[2],
                v[1],
                v[0]
            );
        }
        else
        {
            return _mm256_set_pd(
                0.0,
                v[2],
                v[1],
                v[0]
            );
        }
    }

    template <float_num T>
    inline VecSIMD<T, 3> StoreVector3(__m128 value)
    {
        alignas(16) T data[4];
        _mm_store_ps(data, value);

        return VecSIMD<T, 3>(
            data[0],
            data[1],
            data[2]
        );
    }

    template <float_num T>
    inline VecSIMD<T, 3> StoreVector3(__m256d value)
    {
        alignas(32) T data[4];
        _mm256_store_pd(data, value);

        return VecSIMD<T, 3>(
            data[0],
            data[1],
            data[2]
        );
    }

    template <float_num T>
    inline T Vector3LengthSquared(const VecSIMD<T, 3>& v)
    {
        if constexpr (std::same_as<T, float>)
        {
            const __m128 value = LoadVector3(v);
            const __m128 squared = _mm_mul_ps(value, value);
            const __m128 h1 = _mm_hadd_ps(squared, squared);
            const __m128 h2 = _mm_hadd_ps(h1, h1);

            return _mm_cvtss_f32(h2);
        }
        else
        {
            const __m256d value = LoadVector3(v);
            const __m256d squared = _mm256_mul_pd(value, value);
            const __m256d h = _mm256_hadd_pd(squared, squared);

            const __m128d low = _mm256_castpd256_pd128(h);
            const __m128d high = _mm256_extractf128_pd(h, 1);
            const __m128d sum = _mm_add_pd(low, high);

            return _mm_cvtsd_f64(sum);
        }
    }

    template <float_num T>
    inline VecSIMD<T, 3> NormalizeVector3(const VecSIMD<T, 3>& v)
    {
        const T lengthSquared = Vector3LengthSquared(v);

        if (lengthSquared <= T{ 0 })
            return v;

        const T inverseLength =
            T{ 1 } /
            static_cast<T>(
                std::sqrt(static_cast<double>(lengthSquared))
                );

        if constexpr (std::same_as<T, float>)
        {
            const __m128 value = LoadVector3(v);
            const __m128 scalar = _mm_set1_ps(inverseLength);

            return StoreVector3<T>(
                _mm_mul_ps(value, scalar)
            );
        }
        else
        {
            const __m256d value = LoadVector3(v);
            const __m256d scalar = _mm256_set1_pd(inverseLength);

            return StoreVector3<T>(
                _mm256_mul_pd(value, scalar)
            );
        }
    }

    template <float_num T>
    inline VecSIMD<T, 3> CrossVector3(
        const VecSIMD<T, 3>& a,
        const VecSIMD<T, 3>& b)
    {
        if constexpr (std::same_as<T, float>)
        {
            const __m128 va = LoadVector3(a);
            const __m128 vb = LoadVector3(b);

            const __m128 ayzx =
                _mm_shuffle_ps(
                    va,
                    va,
                    _MM_SHUFFLE(3, 0, 2, 1)
                );

            const __m128 azxy =
                _mm_shuffle_ps(
                    va,
                    va,
                    _MM_SHUFFLE(3, 1, 0, 2)
                );

            const __m128 byzx =
                _mm_shuffle_ps(
                    vb,
                    vb,
                    _MM_SHUFFLE(3, 0, 2, 1)
                );

            const __m128 bzxy =
                _mm_shuffle_ps(
                    vb,
                    vb,
                    _MM_SHUFFLE(3, 1, 0, 2)
                );

            const __m128 result =
                _mm_sub_ps(
                    _mm_mul_ps(ayzx, bzxy),
                    _mm_mul_ps(azxy, byzx)
                );

            return StoreVector3<T>(result);
        }
        else
        {
            const __m256d va = LoadVector3(a);
            const __m256d vb = LoadVector3(b);

            const __m256d ayzx =
                _mm256_permute4x64_pd(
                    va,
                    0xC9
                );

            const __m256d azxy =
                _mm256_permute4x64_pd(
                    va,
                    0xD2
                );

            const __m256d byzx =
                _mm256_permute4x64_pd(
                    vb,
                    0xC9
                );

            const __m256d bzxy =
                _mm256_permute4x64_pd(
                    vb,
                    0xD2
                );

            const __m256d result =
                _mm256_sub_pd(
                    _mm256_mul_pd(ayzx, bzxy),
                    _mm256_mul_pd(azxy, byzx)
                );

            return StoreVector3<T>(result);
        }
    }

}

template <float_num T>
inline constexpr QuaternionSIMD<T>::QuaternionSIMD(
    T x,
    T y,
    T z,
    T w)
    : m_data(x, y, z, w)
{
}


template <float_num T>
template <float_num U>
inline constexpr QuaternionSIMD<T>::QuaternionSIMD(
    const VecSIMD<U, 3>& axis,
    U angleRadians)
{
    *this = FromAxisAngle(axis, angleRadians);
}

template <float_num T>
inline constexpr QuaternionSIMD<T> QuaternionSIMD<T>::Identity()
{
    return QuaternionSIMD(
        T{ 0 },
        T{ 0 },
        T{ 0 },
        T{ 1 }
    );
}

template <float_num T>
template <float_num U>
inline constexpr QuaternionSIMD<T> QuaternionSIMD<T>::FromEuler(
    U pitch,
    U yaw,
    U roll)
{
    const U halfPitch = pitch * U{ 0.5 };
    const U halfYaw = yaw * U{ 0.5 };
    const U halfRoll = roll * U{ 0.5 };

    const U sp = std::sin(halfPitch);
    const U cp = std::cos(halfPitch);
    const U sy = std::sin(halfYaw);
    const U cy = std::cos(halfYaw);
    const U sr = std::sin(halfRoll);
    const U cr = std::cos(halfRoll);

    return QuaternionSIMD(
        static_cast<T>(sp * cy * cr - cp * sy * sr),
        static_cast<T>(cp * sy * cr + sp * cy * sr),
        static_cast<T>(cp * cy * sr - sp * sy * cr),
        static_cast<T>(cp * cy * cr + sp * sy * sr)
    );
}

template <float_num T>
template <float_num U>
inline constexpr QuaternionSIMD<T> QuaternionSIMD<T>::FromAxisAngle(
    const VecSIMD<U, 3>& axis,
    U angleRadians)
{
    const U halfAngle = angleRadians * U{ 0.5 };
    const U s = std::sin(halfAngle);
    const U c = std::cos(halfAngle);

    const U lengthSquared =
        axis[0] * axis[0] +
        axis[1] * axis[1] +
        axis[2] * axis[2];

    if (lengthSquared <= U{ 0 })
        return Identity();

    const U inverseLength =
        U{ 1 } /
        static_cast<U>(
            std::sqrt(static_cast<double>(lengthSquared))
            );

    if constexpr (std::same_as<T, float>)
    {
        const __m128 vAxis =
            _mm_set_ps(
                0.0f,
                static_cast<float>(axis[2]),
                static_cast<float>(axis[1]),
                static_cast<float>(axis[0])
            );

        const __m128 vResult =
            _mm_mul_ps(
                _mm_mul_ps(
                    vAxis,
                    _mm_set1_ps(
                        static_cast<float>(inverseLength)
                    )
                ),
                _mm_set1_ps(
                    static_cast<float>(s)
                )
            );

        const __m128 result =
            _mm_blend_ps(
                vResult,
                _mm_set_ps(
                    static_cast<float>(c),
                    0.0f,
                    0.0f,
                    0.0f
                ),
                0b1000
            );

        QuaternionSIMD<T> output;

        _mm_storeu_ps(
            output.Data(),
            result
        );

        return output;
    }
    else
    {
        const __m256d vAxis =
            _mm256_set_pd(
                0.0,
                static_cast<double>(axis[2]),
                static_cast<double>(axis[1]),
                static_cast<double>(axis[0])
            );

        const __m256d vResult =
            _mm256_mul_pd(
                _mm256_mul_pd(
                    vAxis,
                    _mm256_set1_pd(
                        static_cast<double>(inverseLength)
                    )
                ),
                _mm256_set1_pd(
                    static_cast<double>(s)
                )
            );

        const __m256d result =
            _mm256_blend_pd(
                vResult,
                _mm256_set_pd(
                    static_cast<double>(c),
                    0.0,
                    0.0,
                    0.0
                ),
                0b1000
            );

        QuaternionSIMD<T> output;

        _mm256_storeu_pd(
            output.Data(),
            result
        );

        return output;
    }
}

template <float_num T>
template <float_num U>
inline constexpr QuaternionSIMD<T> QuaternionSIMD<T>::FromToRotation(
    const VecSIMD<U, 3>& from,
    const VecSIMD<U, 3>& to)
{
    const VecSIMD<U, 3> nFrom =
        Detail::NormalizeVector3(from);

    const VecSIMD<U, 3> nTo =
        Detail::NormalizeVector3(to);

    const U cosAngle =
        nFrom[0] * nTo[0] +
        nFrom[1] * nTo[1] +
        nFrom[2] * nTo[2];

    if (cosAngle > U{ 1 } - static_cast<U>(1e-6))
        return Identity();

    if (cosAngle < U{ -1 } + static_cast<U>(1e-6))
    {
        VecSIMD<U, 3> axis =
            Detail::CrossVector3(
                VecSIMD<U, 3>{
            U{ 1 },
                U{ 0 },
                U{ 0 }
        },
                nFrom
            );

        if (
            Detail::Vector3LengthSquared(axis) <
            static_cast<U>(1e-12)
            )
        {
            axis =
                Detail::CrossVector3(
                    VecSIMD<U, 3>{
                U{ 0 },
                    U{ 1 },
                    U{ 0 }
            },
                    nFrom
                );
        }

        axis =
            Detail::NormalizeVector3(axis);

        return FromAxisAngle(
            axis,
            std::numbers::pi_v<U>
        );
    }

    const VecSIMD<U, 3> axis =
        Detail::CrossVector3(
            nFrom,
            nTo
        );

    const U s =
        std::sqrt(
            (U{ 1 } + cosAngle) * U {
        2
    }
        );

    const U inverseS = U{ 1 } / s;

    return QuaternionSIMD(
        static_cast<T>(axis[0] * inverseS),
        static_cast<T>(axis[1] * inverseS),
        static_cast<T>(axis[2] * inverseS),
        static_cast<T>(s * U{ 0.5 })
    ).Normalized();
}

template <float_num T>
template <float_num U>
inline constexpr QuaternionSIMD<T> QuaternionSIMD<T>::LookRotation(
    const VecSIMD<U, 3>& forward,
    const VecSIMD<U, 3>& up)
{
    const VecSIMD<U, 3> f =
        Detail::NormalizeVector3(forward);

    const VecSIMD<U, 3> right =
        Detail::NormalizeVector3(
            Detail::CrossVector3(up, f)
        );

    const VecSIMD<U, 3> trueUp =
        Detail::CrossVector3(f, right);

    const U m00 = right[0];
    const U m01 = trueUp[0];
    const U m02 = f[0];

    const U m10 = right[1];
    const U m11 = trueUp[1];
    const U m12 = f[1];

    const U m20 = right[2];
    const U m21 = trueUp[2];
    const U m22 = f[2];

    const U trace = m00 + m11 + m22;

    if (trace > U{ 0 })
    {
        const U s =
            std::sqrt(trace + U{ 1 }) * U { 2 };

        return QuaternionSIMD(
            static_cast<T>((m21 - m12) / s),
            static_cast<T>((m02 - m20) / s),
            static_cast<T>((m10 - m01) / s),
            static_cast<T>(s * U{ 0.25 })
        );
    }

    if (m00 > m11 && m00 > m22)
    {
        const U s =
            std::sqrt(
                U{ 1 } + m00 - m11 - m22
            ) * U {
            2
        };

        return QuaternionSIMD(
            static_cast<T>(s * U{ 0.25 }),
            static_cast<T>((m01 + m10) / s),
            static_cast<T>((m02 + m20) / s),
            static_cast<T>((m21 - m12) / s)
        );
    }

    if (m11 > m22)
    {
        const U s =
            std::sqrt(
                U{ 1 } + m11 - m00 - m22
            ) * U {
            2
        };

        return QuaternionSIMD(
            static_cast<T>((m01 + m10) / s),
            static_cast<T>(s * U{ 0.25 }),
            static_cast<T>((m12 + m21) / s),
            static_cast<T>((m02 - m20) / s)
        );
    }

    const U s =
        std::sqrt(
            U{ 1 } + m22 - m00 - m11
        ) * U {
        2
    };

    return QuaternionSIMD(
        static_cast<T>((m02 + m20) / s),
        static_cast<T>((m12 + m21) / s),
        static_cast<T>(s * U{ 0.25 }),
        static_cast<T>((m10 - m01) / s)
    );
}

template <float_num T>
inline constexpr T& QuaternionSIMD<T>::X()
{
    return m_data[0];
}

template <float_num T>
inline constexpr const T& QuaternionSIMD<T>::X() const
{
    return m_data[0];
}

template <float_num T>
inline constexpr T& QuaternionSIMD<T>::Y()
{
    return m_data[1];
}

template <float_num T>
inline constexpr const T& QuaternionSIMD<T>::Y() const
{
    return m_data[1];
}

template <float_num T>
inline constexpr T& QuaternionSIMD<T>::Z()
{
    return m_data[2];
}

template <float_num T>
inline constexpr const T& QuaternionSIMD<T>::Z() const
{
    return m_data[2];
}

template <float_num T>
inline constexpr T& QuaternionSIMD<T>::W()
{
    return m_data[3];
}

template <float_num T>
inline constexpr const T& QuaternionSIMD<T>::W() const
{
    return m_data[3];
}

template <float_num T>
inline constexpr T* QuaternionSIMD<T>::Data()
{
    return m_data.Data();
}

template <float_num T>
inline constexpr const T* QuaternionSIMD<T>::Data() const
{
    return m_data.Data();
}

template <float_num T>
inline constexpr std::size_t QuaternionSIMD<T>::Size()
{
    return 4;
}

template <float_num T>
template <float_num U>
inline constexpr bool QuaternionSIMD<T>::operator==(
    const QuaternionSIMD<U>& other) const
{
    if constexpr (std::same_as<T, U>)
    {
        if constexpr (std::same_as<T, float>)
        {
            const __m128 a = _mm_loadu_ps(Data());
            const __m128 b = _mm_loadu_ps(other.Data());

            return _mm_movemask_ps(
                _mm_cmpeq_ps(a, b)
            ) == 0xF;
        }
        else
        {
            const __m256d a = _mm256_loadu_pd(Data());
            const __m256d b = _mm256_loadu_pd(other.Data());

            return _mm256_movemask_pd(
                _mm256_cmp_pd(
                    a,
                    b,
                    _CMP_EQ_OQ
                )
            ) == 0xF;
        }
    }
    else
    {
        return
            X() == static_cast<T>(other.X()) &&
            Y() == static_cast<T>(other.Y()) &&
            Z() == static_cast<T>(other.Z()) &&
            W() == static_cast<T>(other.W());
    }
}

template <float_num T>
template <float_num U>
inline constexpr QuaternionSIMD<T>& QuaternionSIMD<T>::operator+=(
    const QuaternionSIMD<U>& other)
{
    if constexpr (std::same_as<T, U>)
    {
        if constexpr (std::same_as<T, float>)
        {
            _mm_storeu_ps(
                Data(),
                _mm_add_ps(
                    _mm_loadu_ps(Data()),
                    _mm_loadu_ps(other.Data())
                )
            );
        }
        else
        {
            _mm256_storeu_pd(
                Data(),
                _mm256_add_pd(
                    _mm256_loadu_pd(Data()),
                    _mm256_loadu_pd(other.Data())
                )
            );
        }
    }
    else
    {
        m_data[0] += static_cast<T>(other.X());
        m_data[1] += static_cast<T>(other.Y());
        m_data[2] += static_cast<T>(other.Z());
        m_data[3] += static_cast<T>(other.W());
    }

    return *this;
}

template <float_num T>
template <float_num U>
inline constexpr QuaternionSIMD<T>& QuaternionSIMD<T>::operator-=(
    const QuaternionSIMD<U>& other)
{
    if constexpr (std::same_as<T, U>)
    {
        if constexpr (std::same_as<T, float>)
        {
            _mm_storeu_ps(
                Data(),
                _mm_sub_ps(
                    _mm_loadu_ps(Data()),
                    _mm_loadu_ps(other.Data())
                )
            );
        }
        else
        {
            _mm256_storeu_pd(
                Data(),
                _mm256_sub_pd(
                    _mm256_loadu_pd(Data()),
                    _mm256_loadu_pd(other.Data())
                )
            );
        }
    }
    else
    {
        m_data[0] -= static_cast<T>(other.X());
        m_data[1] -= static_cast<T>(other.Y());
        m_data[2] -= static_cast<T>(other.Z());
        m_data[3] -= static_cast<T>(other.W());
    }

    return *this;
}

template <float_num T>
template <float_num U>
inline constexpr QuaternionSIMD<T>& QuaternionSIMD<T>::operator*=(
    const QuaternionSIMD<U>& other)
{
    if constexpr (!std::same_as<T, U>)
    {
        const T x =
            W() * static_cast<T>(other.X()) +
            X() * static_cast<T>(other.W()) +
            Y() * static_cast<T>(other.Z()) -
            Z() * static_cast<T>(other.Y());

        const T y =
            W() * static_cast<T>(other.Y()) -
            X() * static_cast<T>(other.Z()) +
            Y() * static_cast<T>(other.W()) +
            Z() * static_cast<T>(other.X());

        const T z =
            W() * static_cast<T>(other.Z()) +
            X() * static_cast<T>(other.Y()) -
            Y() * static_cast<T>(other.X()) +
            Z() * static_cast<T>(other.W());

        const T w =
            W() * static_cast<T>(other.W()) -
            X() * static_cast<T>(other.X()) -
            Y() * static_cast<T>(other.Y()) -
            Z() * static_cast<T>(other.Z());

        m_data[0] = x;
        m_data[1] = y;
        m_data[2] = z;
        m_data[3] = w;

        return *this;
    }

    if constexpr (std::same_as<T, float>)
    {
        const __m128 a = _mm_loadu_ps(Data());
        const __m128 b = _mm_loadu_ps(other.Data());

        const __m128 zero = _mm_setzero_ps();

        const __m128 av =
            _mm_blend_ps(a, zero, 0b1000);

        const __m128 bv =
            _mm_blend_ps(b, zero, 0b1000);

        const __m128 ayzx =
            _mm_shuffle_ps(
                av,
                av,
                _MM_SHUFFLE(3, 0, 2, 1)
            );

        const __m128 azxy =
            _mm_shuffle_ps(
                av,
                av,
                _MM_SHUFFLE(3, 1, 0, 2)
            );

        const __m128 byzx =
            _mm_shuffle_ps(
                bv,
                bv,
                _MM_SHUFFLE(3, 0, 2, 1)
            );

        const __m128 bzxy =
            _mm_shuffle_ps(
                bv,
                bv,
                _MM_SHUFFLE(3, 1, 0, 2)
            );

        const __m128 cross =
            _mm_sub_ps(
                _mm_mul_ps(ayzx, bzxy),
                _mm_mul_ps(azxy, byzx)
            );

        const __m128 aw =
            _mm_shuffle_ps(
                a,
                a,
                _MM_SHUFFLE(3, 3, 3, 3)
            );

        const __m128 bw =
            _mm_shuffle_ps(
                b,
                b,
                _MM_SHUFFLE(3, 3, 3, 3)
            );

        const __m128 vectorResult =
            _mm_add_ps(
                _mm_add_ps(
                    cross,
                    _mm_mul_ps(aw, bv)
                ),
                _mm_mul_ps(bw, av)
            );

        const __m128 mul =
            _mm_mul_ps(av, bv);

        const __m128 h1 =
            _mm_hadd_ps(mul, mul);

        const __m128 h2 =
            _mm_hadd_ps(h1, h1);

        const __m128 w =
            _mm_sub_ps(
                _mm_mul_ps(aw, bw),
                h2
            );

        _mm_storeu_ps(
            Data(),
            _mm_blend_ps(
                vectorResult,
                w,
                0b1000
            )
        );
    }
    else
    {
        const __m256d a = _mm256_loadu_pd(Data());
        const __m256d b = _mm256_loadu_pd(other.Data());

        const __m256d zero = _mm256_setzero_pd();

        const __m256d av =
            _mm256_blend_pd(a, zero, 0b1000);

        const __m256d bv =
            _mm256_blend_pd(b, zero, 0b1000);

        const __m256d ayzx =
            _mm256_permute4x64_pd(av, 0xC9);

        const __m256d azxy =
            _mm256_permute4x64_pd(av, 0xD2);

        const __m256d byzx =
            _mm256_permute4x64_pd(bv, 0xC9);

        const __m256d bzxy =
            _mm256_permute4x64_pd(bv, 0xD2);

        const __m256d cross =
            _mm256_sub_pd(
                _mm256_mul_pd(ayzx, bzxy),
                _mm256_mul_pd(azxy, byzx)
            );

        const __m256d aw =
            _mm256_permute4x64_pd(a, 0xFF);

        const __m256d bw =
            _mm256_permute4x64_pd(b, 0xFF);

        const __m256d vectorResult =
            _mm256_add_pd(
                _mm256_add_pd(
                    cross,
                    _mm256_mul_pd(aw, bv)
                ),
                _mm256_mul_pd(bw, av)
            );

        const __m256d mul =
            _mm256_mul_pd(av, bv);

        const __m256d h =
            _mm256_hadd_pd(mul, mul);

        const __m128d low =
            _mm256_castpd256_pd128(h);

        const __m128d high =
            _mm256_extractf128_pd(h, 1);

        const __m128d sum =
            _mm_add_pd(low, high);

        const double dot =
            _mm_cvtsd_f64(sum);

        const __m256d w =
            _mm256_sub_pd(
                _mm256_mul_pd(aw, bw),
                _mm256_set1_pd(dot)
            );

        _mm256_storeu_pd(
            Data(),
            _mm256_blend_pd(
                vectorResult,
                w,
                0b1000
            )
        );
    }

    return *this;
}

template <float_num T>
template <float_num U>
inline constexpr QuaternionSIMD<T>& QuaternionSIMD<T>::operator*=(U scalar)
{
    if constexpr (std::same_as<T, float>)
    {
        _mm_storeu_ps(
            Data(),
            _mm_mul_ps(
                _mm_loadu_ps(Data()),
                _mm_set1_ps(static_cast<float>(scalar))
            )
        );
    }
    else
    {
        _mm256_storeu_pd(
            Data(),
            _mm256_mul_pd(
                _mm256_loadu_pd(Data()),
                _mm256_set1_pd(static_cast<double>(scalar))
            )
        );
    }

    return *this;
}

template <float_num T>
template <float_num U>
inline constexpr QuaternionSIMD<T>& QuaternionSIMD<T>::operator/=(U scalar)
{
    if constexpr (std::same_as<T, float>)
    {
        _mm_storeu_ps(
            Data(),
            _mm_div_ps(
                _mm_loadu_ps(Data()),
                _mm_set1_ps(static_cast<float>(scalar))
            )
        );
    }
    else
    {
        _mm256_storeu_pd(
            Data(),
            _mm256_div_pd(
                _mm256_loadu_pd(Data()),
                _mm256_set1_pd(static_cast<double>(scalar))
            )
        );
    }

    return *this;
}

template <float_num T>
inline constexpr QuaternionSIMD<T> QuaternionSIMD<T>::operator+() const
{
    return *this;
}

template <float_num T>
inline constexpr QuaternionSIMD<T> QuaternionSIMD<T>::operator-() const
{
    QuaternionSIMD result;

    if constexpr (std::same_as<T, float>)
    {
        _mm_storeu_ps(
            result.Data(),
            _mm_sub_ps(
                _mm_setzero_ps(),
                _mm_loadu_ps(Data())
            )
        );
    }
    else
    {
        _mm256_storeu_pd(
            result.Data(),
            _mm256_sub_pd(
                _mm256_setzero_pd(),
                _mm256_loadu_pd(Data())
            )
        );
    }

    return result;
}

template <float_num T>
inline constexpr T QuaternionSIMD<T>::LengthSquared() const
{
    return Detail::HorizontalSumSquared(Data());
}

template <float_num T>
inline constexpr T QuaternionSIMD<T>::Length() const
{
    return static_cast<T>(
        std::sqrt(
            static_cast<double>(LengthSquared())
        )
        );
}

template <float_num T>
inline constexpr QuaternionSIMD<T> QuaternionSIMD<T>::Normalized() const
{
    const T length = Length();

    if (length <= T{ 0 })
        return *this;

    QuaternionSIMD result = *this;
    result /= length;

    return result;
}

template <float_num T>
inline constexpr void QuaternionSIMD<T>::Normalize()
{
    *this = Normalized();
}

template <float_num T>
inline constexpr QuaternionSIMD<T> QuaternionSIMD<T>::Conjugate() const
{
    QuaternionSIMD result;

    if constexpr (std::same_as<T, float>)
    {
        _mm_storeu_ps(
            result.Data(),
            _mm_mul_ps(
                _mm_loadu_ps(Data()),
                _mm_set_ps(
                    1.0f,
                    -1.0f,
                    -1.0f,
                    -1.0f
                )
            )
        );
    }
    else
    {
        _mm256_storeu_pd(
            result.Data(),
            _mm256_mul_pd(
                _mm256_loadu_pd(Data()),
                _mm256_set_pd(
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0
                )
            )
        );
    }

    return result;
}

template <float_num T>
inline constexpr QuaternionSIMD<T> QuaternionSIMD<T>::Inverse() const
{
    const T lengthSquared = LengthSquared();

    if (lengthSquared <= T{ 0 })
        return Conjugate();

    return Conjugate() / lengthSquared;
}

template <float_num T>
inline constexpr VecSIMD<T, 3> QuaternionSIMD<T>::RotateVector(
    const VecSIMD<T, 3>& v) const
{
    const VecSIMD<T, 3> qVector{
        X(),
        Y(),
        Z()
    };

    const VecSIMD<T, 3> cross1 =
        Detail::CrossVector3(
            qVector,
            v
        );

    const VecSIMD<T, 3> cross2 =
        Detail::CrossVector3(
            qVector,
            cross1
        );

    if constexpr (std::same_as<T, float>)
    {
        const __m128 result =
            _mm_add_ps(
                Detail::LoadVector3(v),
                _mm_add_ps(
                    _mm_mul_ps(
                        Detail::LoadVector3(cross1),
                        _mm_set1_ps(2.0f * W())
                    ),
                    _mm_mul_ps(
                        Detail::LoadVector3(cross2),
                        _mm_set1_ps(2.0f)
                    )
                )
            );

        return Detail::StoreVector3<T>(result);
    }
    else
    {
        const __m256d result =
            _mm256_add_pd(
                Detail::LoadVector3(v),
                _mm256_add_pd(
                    _mm256_mul_pd(
                        Detail::LoadVector3(cross1),
                        _mm256_set1_pd(2.0 * W())
                    ),
                    _mm256_mul_pd(
                        Detail::LoadVector3(cross2),
                        _mm256_set1_pd(2.0)
                    )
                )
            );

        return Detail::StoreVector3<T>(result);
    }
}

template <float_num T>
inline constexpr VecSIMD<T, 3> QuaternionSIMD<T>::ToEuler() const
{
    return VecSIMD<T, 3>{
        Pitch(),
            Yaw(),
            Roll()
    };
}

template <float_num T>
inline constexpr T QuaternionSIMD<T>::Pitch() const
{
    const T sinPitch =
        T{ 2 } *(
            W() * X() +
            Y() * Z()
            );

    const T cosPitch =
        T{ 1 } -
        T{ 2 } *(
            X() * X() +
            Y() * Y()
            );

    return std::atan2(
        sinPitch,
        cosPitch
    );
}

template <float_num T>
inline constexpr T QuaternionSIMD<T>::Yaw() const
{
    const T sinYaw =
        T{ 2 } *(
            W() * Y() -
            Z() * X()
            );

    if (std::abs(sinYaw) >= T{ 1 })
    {
        return std::copysign(
            std::numbers::pi_v<T> / T{ 2 },
            sinYaw
        );
    }

    return std::asin(sinYaw);
}

template <float_num T>
inline constexpr T QuaternionSIMD<T>::Roll() const
{
    const T sinRoll =
        T{ 2 } *(
            W() * Z() +
            X() * Y()
            );

    const T cosRoll =
        T{ 1 } -
        T{ 2 } *(
            Y() * Y() +
            Z() * Z()
            );

    return std::atan2(
        sinRoll,
        cosRoll
    );
}

template <float_num T>
inline constexpr void QuaternionSIMD<T>::ToAxisAngle(
    VecSIMD<T, 3>& outAxis,
    T& outAngleRadians) const
{
    const QuaternionSIMD normalized =
        Normalized();

    const T clampedW =
        normalized.W() < T{ -1 }
        ? T{ -1 }
        : (
            normalized.W() > T{ 1 }
            ? T{ 1 }
            : normalized.W()
            );

    outAngleRadians =
        T{ 2 } *
        std::acos(clampedW);

    const T s =
        std::sqrt(
            T{ 1 } -
            clampedW * clampedW
        );

    if (s < static_cast<T>(1e-6))
    {
        outAxis =
            VecSIMD<T, 3>{
                T{ 1 },
                T{ 0 },
                T{ 0 }
        };
    }
    else
    {
        outAxis =
            VecSIMD<T, 3>{
                normalized.X() / s,
                normalized.Y() / s,
                normalized.Z() / s
        };
    }
}

template <float_num T, float_num U>
constexpr auto operator+(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<U>& b
    ) -> QuaternionCommonSIMD<T, U>
{
    QuaternionCommonSIMD<T, U> result;

    if constexpr (!std::same_as<T, U>)
    {
        if constexpr (std::same_as<T, float>)
        {
            const __m256d va = _mm256_cvtps_pd(_mm_loadu_ps(a.Data()));
            const __m256d vb = _mm256_loadu_pd(b.Data());

            _mm256_storeu_pd(result.Data(), _mm256_add_pd(va, vb));
        }
        else
        {
            const __m256d va = _mm256_loadu_pd(a.Data());
            const __m256d vb = _mm256_cvtps_pd(_mm_loadu_ps(b.Data()));

            _mm256_storeu_pd(result.Data(), _mm256_add_pd(va, vb));
        }

        return result;
    }

    if constexpr (std::same_as<T, float> && std::same_as<U, float>)
    {
        const __m128 va = _mm_loadu_ps(a.Data());
        const __m128 vb = _mm_loadu_ps(b.Data());

        _mm_storeu_ps(result.Data(), _mm_add_ps(va, vb));
    }
    else if constexpr (std::same_as<T, double> && std::same_as<U, double>)
    {
        const __m256d va = _mm256_loadu_pd(a.Data());
        const __m256d vb = _mm256_loadu_pd(b.Data());

        _mm256_storeu_pd(result.Data(), _mm256_add_pd(va, vb));
    }

    return result;
}

template <float_num T, float_num U>
constexpr auto operator-(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<U>& b
    ) -> QuaternionCommonSIMD<T, U>
{
    QuaternionCommonSIMD<T, U> result;

    if constexpr (!std::same_as<T, U>)
    {
        if constexpr (std::same_as<T, float>)
        {
            const __m256d va = _mm256_cvtps_pd(_mm_loadu_ps(a.Data()));
            const __m256d vb = _mm256_loadu_pd(b.Data());

            _mm256_storeu_pd(result.Data(), _mm256_sub_pd(va, vb));
        }
        else
        {
            const __m256d va = _mm256_loadu_pd(a.Data());
            const __m256d vb = _mm256_cvtps_pd(_mm_loadu_ps(b.Data()));

            _mm256_storeu_pd(result.Data(), _mm256_sub_pd(va, vb));
        }

        return result;
    }

    if constexpr (std::same_as<T, float> && std::same_as<U, float>)
    {
        const __m128 va = _mm_loadu_ps(a.Data());
        const __m128 vb = _mm_loadu_ps(b.Data());

        _mm_storeu_ps(result.Data(), _mm_sub_ps(va, vb));
    }
    else if constexpr (std::same_as<T, double> && std::same_as<U, double>)
    {
        const __m256d va = _mm256_loadu_pd(a.Data());
        const __m256d vb = _mm256_loadu_pd(b.Data());

        _mm256_storeu_pd(result.Data(), _mm256_sub_pd(va, vb));
    }

    return result;
}

template <typename Tr, float_num S>
static inline auto LoadQuat(const QuaternionSIMD<S>& q)
{
    if constexpr (std::same_as<typename Tr::type, __m256d> && std::same_as<S, float>)
        return _mm256_cvtps_pd(simd_traits<float, 128>::load(q.Data()));
    else
        return Tr::load(q.Data());
}

template <float_num T, float_num U>
inline constexpr auto operator*(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<U>& b)
    -> QuaternionCommonSIMD<T, U>
{
    using R = std::common_type_t<T, U>;
    using Tr = simd_traits<R, std::same_as<R, float> ? 128 : 256>;

    QuaternionCommonSIMD<T, U> result;

    const auto va = LoadQuat<Tr>(a);
    const auto vb = LoadQuat<Tr>(b);

    // parties vectorielles (w mis à 0)
    const auto zero = Tr::set1(R{ 0 });
    const auto av = Tr::template blend<0b1000>(va, zero);
    const auto bv = Tr::template blend<0b1000>(vb, zero);

    const auto aw = Tr::template permute<3, 3, 3, 3>(va);
    const auto bw = Tr::template permute<3, 3, 3, 3>(vb);

    // a.v x b.v = a.yzx * b.zxy - a.zxy * b.yzx
    const auto cross = Tr::sub(
        Tr::mul(Tr::template permute<1, 2, 0, 3>(av), Tr::template permute<2, 0, 1, 3>(bv)),
        Tr::mul(Tr::template permute<2, 0, 1, 3>(av), Tr::template permute<1, 2, 0, 3>(bv)));

    // v = aw*bv + bw*av + cross
    const auto v = Tr::add(cross, Tr::add(Tr::mul(aw, bv), Tr::mul(bw, av)));

    // w = aw*bw - dot(av, bv)
    const auto w = Tr::sub(Tr::mul(aw, bw), Tr::set1(Tr::reduce_add(Tr::mul(av, bv))));

    Tr::store(result.Data(), Tr::template blend<0b1000>(v, w));
    return result;
}

template <float_num T, float_num U>
constexpr auto operator*(
    const QuaternionSIMD<T>& q,
    U scalar
    ) -> QuaternionCommonSIMD<T, U>
{
    QuaternionCommonSIMD<T, U> result;

    if constexpr (!std::same_as<T, U>)
    {
        if constexpr (std::same_as<T, float>)
        {
            const __m256d va = _mm256_cvtps_pd(_mm_loadu_ps(q.Data()));
            const __m256d vb = _mm256_set1_pd(static_cast<double>(scalar));

            _mm256_storeu_pd(result.Data(), _mm256_mul_pd(va, vb));
        }
        else
        {
            const __m256d va = _mm256_loadu_pd(q.Data());
            const __m256d vb = _mm256_set1_pd(static_cast<double>(scalar));

            _mm256_storeu_pd(result.Data(), _mm256_mul_pd(va, vb));
        }

        return result;
    }

    if constexpr (std::same_as<T, float> && std::same_as<U, float>)
    {
        const __m128 v = _mm_loadu_ps(q.Data());
        const __m128 s = _mm_set1_ps(static_cast<float>(scalar));

        _mm_storeu_ps(result.Data(), _mm_mul_ps(v, s));
    }
    else if constexpr (std::same_as<T, double> && std::same_as<U, double>)
    {
        const __m256d v = _mm256_loadu_pd(q.Data());
        const __m256d s = _mm256_set1_pd(static_cast<double>(scalar));

        _mm256_storeu_pd(result.Data(), _mm256_mul_pd(v, s));
    }

    return result;
}

template <float_num T, float_num U>
inline constexpr auto operator*(
    U scalar,
    const QuaternionSIMD<T>& q)
    -> QuaternionCommonSIMD<T, U>
{
    return q * scalar;
}

template <float_num T, float_num U>
inline constexpr auto operator/(
    const QuaternionSIMD<T>& q,
    U scalar)
    -> QuaternionCommonSIMD<T, U>
{
    using C = std::common_type_t<T, U>;
    using Tr = simd_traits<C, (sizeof(C) == sizeof(float)) ? 128 : 256>;

    QuaternionCommonSIMD<T, U> result;

    typename Tr::type v;
    if constexpr (std::same_as<T, C>)
    {
        v = Tr::load(q.Data());
    }
    else // T = float, C = double : on élargit les 4 floats en 4 doubles
    {
        v = _mm256_cvtps_pd(simd_traits<float, 128>::load(q.Data()));
    }

    Tr::store(result.Data(), Tr::div(v, Tr::set1(static_cast<C>(scalar))));
    return result;
}

template <float_num T, float_num U>
inline constexpr auto operator*(
    const QuaternionSIMD<T>& q,
    const VecSIMD<U, 3>& v)
    -> VecSIMD<std::common_type_t<T, U>, 3>
{
    using R = std::common_type_t<T, U>;

    const QuaternionSIMD<R> quaternion{
        static_cast<R>(q.X()),
        static_cast<R>(q.Y()),
        static_cast<R>(q.Z()),
        static_cast<R>(q.W())
    };

    const VecSIMD<R, 3> vector{
        static_cast<R>(v[0]),
        static_cast<R>(v[1]),
        static_cast<R>(v[2])
    };

    return quaternion.RotateVector(vector);
}

template <float_num T>
inline constexpr T Dot(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<T>& b)
{
    if constexpr (std::same_as<T, float>)
    {
        const __m128 mul =
            _mm_mul_ps(
                _mm_loadu_ps(a.Data()),
                _mm_loadu_ps(b.Data())
            );

        const __m128 h1 =
            _mm_hadd_ps(mul, mul);

        const __m128 h2 =
            _mm_hadd_ps(h1, h1);

        return _mm_cvtss_f32(h2);
    }
    else
    {
        const __m256d mul =
            _mm256_mul_pd(
                _mm256_loadu_pd(a.Data()),
                _mm256_loadu_pd(b.Data())
            );

        const __m256d h =
            _mm256_hadd_pd(mul, mul);

        const __m128d low =
            _mm256_castpd256_pd128(h);

        const __m128d high =
            _mm256_extractf128_pd(h, 1);

        return _mm_cvtsd_f64(
            _mm_add_pd(low, high)
        );
    }
}

template <float_num T>
inline constexpr QuaternionSIMD<T> LerpUnclamped(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<T>& b,
    T t)
{
    return (
        a * (T{ 1 } - t) +
        b * t
        ).Normalized();
}

template <float_num T>
inline constexpr QuaternionSIMD<T> Lerp(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<T>& b,
    T t)
{
    const T clampedT =
        t < T{ 0 }
        ? T{ 0 }
        : (
            t > T{ 1 }
            ? T{ 1 }
            : t
            );

    return LerpUnclamped(
        a,
        b,
        clampedT
    );
}

template <float_num T>
inline constexpr QuaternionSIMD<T> SlerpUnclamped(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<T>& b,
    T t)
{
    QuaternionSIMD<T> end = b;
    T cosOmega = Dot(a, b);

    if (cosOmega < T{ 0 })
    {
        cosOmega = -cosOmega;
        end = -end;
    }

    constexpr T kEpsilon =
        static_cast<T>(1e-6);

    if (cosOmega > T{ 1 } - kEpsilon)
        return LerpUnclamped(a, end, t);

    const T omega = std::acos(cosOmega);
    const T sinOmega = std::sin(omega);

    const T scaleA =
        std::sin(
            (T{ 1 } - t) * omega
        ) / sinOmega;

    const T scaleB =
        std::sin(t * omega) / sinOmega;

    return a * scaleA + end * scaleB;
}

template <float_num T>
inline constexpr QuaternionSIMD<T> Slerp(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<T>& b,
    T t)
{
    const T clampedT =
        t < T{ 0 }
        ? T{ 0 }
        : (
            t > T{ 1 }
            ? T{ 1 }
            : t
            );

    return SlerpUnclamped(
        a,
        b,
        clampedT
    );
}

template <float_num T>
inline constexpr T Angle(
    const QuaternionSIMD<T>& a,
    const QuaternionSIMD<T>& b)
{
    const T cosHalfAngle = Dot(a, b);

    const T clamped =
        cosHalfAngle < T{ -1 }
        ? T{ -1 }
        : (
            cosHalfAngle > T{ 1 }
            ? T{ 1 }
            : cosHalfAngle
            );

    return T{ 2 } *
        std::acos(
            std::abs(clamped)
        );
}

template <float_num T>
inline constexpr QuaternionSIMD<T> RotateTowards(
    const QuaternionSIMD<T>& from,
    const QuaternionSIMD<T>& to,
    T maxAngleRadians)
{
    const T angle = Angle(from, to);

    if (angle <= T{ 0 })
        return to;

    const T t =
        maxAngleRadians / angle;

    if (t >= T{ 1 })
        return to;

    return SlerpUnclamped(
        from,
        to,
        t
    );
}

template <float_num T>
struct std::formatter<QuaternionSIMD<T>>
{
    std::formatter<T> underlying;

    constexpr auto parse(
        std::format_parse_context& ctx)
    {
        return underlying.parse(ctx);
    }

    auto format(
        const QuaternionSIMD<T>& obj,
        std::format_context& ctx) const
    {
        auto out = ctx.out();

        out = std::format_to(out, "(");
        out = underlying.format(obj.X(), ctx);
        out = std::format_to(out, ", ");
        out = underlying.format(obj.Y(), ctx);
        out = std::format_to(out, ", ");
        out = underlying.format(obj.Z(), ctx);
        out = std::format_to(out, ", ");
        out = underlying.format(obj.W(), ctx);

        return std::format_to(out, ")");
    }
};