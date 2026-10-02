#pragma once

template <float_num T, std::size_t N>
template <typename... Args>
    requires(sizeof...(Args) == N) &&
(std::convertible_to<Args, T> && ...)
inline constexpr VecSIMD<T, N>::VecSIMD(Args&&... args)
    : m_data{ static_cast<T>(std::forward<Args>(args))... }
{
}

template <float_num T, std::size_t N>
inline constexpr T& VecSIMD<T, N>::operator[](size_type index)
{
    return m_data[index];
}

template <float_num T, std::size_t N>
inline constexpr const T& VecSIMD<T, N>::operator[](size_type index) const
{
    return m_data[index];
}

template <float_num T, std::size_t N>
template <std::size_t I>
inline auto VecSIMD<T, N>::get_m128() const
{
    using traits = simd_traits<T, 128>;

    static_assert((I + 1) * traits::width <= N);

    return traits::load(m_data.data() + I * traits::width);
}

template <float_num T, std::size_t N>
template <std::size_t I>
inline auto VecSIMD<T, N>::get_m256() const
{
    using traits = simd_traits<T, 256>;

    static_assert((I + 1) * traits::width <= N);

    return traits::load(m_data.data() + I * traits::width);
}

template <float_num T, std::size_t N>
inline constexpr T* VecSIMD<T, N>::Data()
{
    return m_data.data();
}

template <float_num T, std::size_t N>
inline constexpr const T* VecSIMD<T, N>::Data() const
{
    return m_data.data();
}

template <float_num T, std::size_t N>
inline constexpr std::size_t VecSIMD<T, N>::Size()
{
    return N;
}

template <float_num T, std::size_t N>
template <float_num U>
inline constexpr bool VecSIMD<T, N>::operator==(
    const VecSIMD<U, N>& other
    ) const
{
    for (std::size_t i = 0; i < N; ++i)
    {
        if ((*this)[i] != other[i])
            return false;
    }

    return true;
}

template <float_num T, std::size_t N>
template <float_num U>
inline constexpr VecSIMD<T, N>& VecSIMD<T, N>::operator+=(
    const VecSIMD<U, N>& other
    )
{
    *this = *this + other;
    return *this;
}

template <float_num T, std::size_t N>
template <float_num U>
inline constexpr VecSIMD<T, N>& VecSIMD<T, N>::operator-=(
    const VecSIMD<U, N>& other
    )
{
    *this = *this - other;
    return *this;
}

template <float_num T, std::size_t N>
template <float_num U>
inline constexpr VecSIMD<T, N>& VecSIMD<T, N>::operator*=(
    const VecSIMD<U, N>& other
    )
{
    *this = *this * other;
    return *this;
}

template <float_num T, std::size_t N>
template <float_num U>
inline constexpr VecSIMD<T, N>& VecSIMD<T, N>::operator/=(
    const VecSIMD<U, N>& other
    )
{
    *this = *this / other;
    return *this;
}

template <float_num T, std::size_t N>
template <float_num U>
inline constexpr VecSIMD<T, N>& VecSIMD<T, N>::operator*=(U scalar)
{
    *this = *this * scalar;
    return *this;
}

template <float_num T, std::size_t N>
template <float_num U>
inline constexpr VecSIMD<T, N>& VecSIMD<T, N>::operator/=(U scalar)
{
    *this = *this / scalar;
    return *this;
}

template <float_num T, float_num U, std::size_t N>
inline auto operator+(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
    ) -> VecSIMDCommon<T, U, N>
{
    using R = std::common_type_t<T, U>;

    VecSIMDCommon<T, U, N> result;

    if constexpr (std::same_as<R, float> && N == 4)
    {
        const __m128 va = _mm_loadu_ps(a.Data());
        const __m128 vb = _mm_loadu_ps(b.Data());

        _mm_storeu_ps(result.Data(), _mm_add_ps(va, vb));
    }
    else if constexpr (std::same_as<R, float> && N == 8)
    {
        const __m256 va = _mm256_loadu_ps(a.Data());
        const __m256 vb = _mm256_loadu_ps(b.Data());

        _mm256_storeu_ps(result.Data(), _mm256_add_ps(va, vb));
    }
    else if constexpr (std::same_as<R, double> && N == 2)
    {
        const __m128d va = _mm_loadu_pd(a.Data());
        const __m128d vb = _mm_loadu_pd(b.Data());

        _mm_storeu_pd(result.Data(), _mm_add_pd(va, vb));
    }
    else if constexpr (std::same_as<R, double> && N == 4)
    {
        const __m256d va = _mm256_loadu_pd(a.Data());
        const __m256d vb = _mm256_loadu_pd(b.Data());

        _mm256_storeu_pd(result.Data(), _mm256_add_pd(va, vb));
    }
    else {
        for (std::size_t i = 0; i < N; ++i) {
            result[i] = static_cast<R>(a[i]) + static_cast<R>(b[i]);
        }
    }

    return result;
}

template <float_num T, float_num U, std::size_t N>
inline auto operator-(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
    ) -> VecSIMDCommon<T, U, N>
{
    using R = std::common_type_t<T, U>;

    VecSIMDCommon<T, U, N> result;

    if constexpr (std::same_as<R, float> && N == 4)
    {
        const __m128 va = _mm_loadu_ps(a.Data());
        const __m128 vb = _mm_loadu_ps(b.Data());

        _mm_storeu_ps(result.Data(), _mm_sub_ps(va, vb));
    }
    else if constexpr (std::same_as<R, float> && N == 8)
    {
        const __m256 va = _mm256_loadu_ps(a.Data());
        const __m256 vb = _mm256_loadu_ps(b.Data());

        _mm256_storeu_ps(result.Data(), _mm256_sub_ps(va, vb));
    }
    else if constexpr (std::same_as<R, double> && N == 2)
    {
        const __m128d va = _mm_loadu_pd(a.Data());
        const __m128d vb = _mm_loadu_pd(b.Data());

        _mm_storeu_pd(result.Data(), _mm_sub_pd(va, vb));
    }
    else if constexpr (std::same_as<R, double> && N == 4)
    {
        const __m256d va = _mm256_loadu_pd(a.Data());
        const __m256d vb = _mm256_loadu_pd(b.Data());

        _mm256_storeu_pd(result.Data(), _mm256_sub_pd(va, vb));
    }
    else {
        for (std::size_t i = 0; i < N; ++i) {
            result[i] = static_cast<R>(a[i]) - static_cast<R>(b[i]);
        }
    }

    return result;
}

template <float_num T, float_num U, std::size_t N>
inline auto operator*(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
    ) -> VecSIMDCommon<T, U, N>
{
    using R = std::common_type_t<T, U>;

    VecSIMDCommon<T, U, N> result;

    if constexpr (std::same_as<R, float> && N == 4)
    {
        const __m128 va = _mm_loadu_ps(a.Data());
        const __m128 vb = _mm_loadu_ps(b.Data());

        _mm_storeu_ps(result.Data(), _mm_mul_ps(va, vb));
    }
    else if constexpr (std::same_as<R, float> && N == 8)
    {
        const __m256 va = _mm256_loadu_ps(a.Data());
        const __m256 vb = _mm256_loadu_ps(b.Data());

        _mm256_storeu_ps(result.Data(), _mm256_mul_ps(va, vb));
    }
    else if constexpr (std::same_as<R, double> && N == 2)
    {
        const __m128d va = _mm_loadu_pd(a.Data());
        const __m128d vb = _mm_loadu_pd(b.Data());

        _mm_storeu_pd(result.Data(), _mm_mul_pd(va, vb));
    }
    else if constexpr (std::same_as<R, double> && N == 4)
    {
        const __m256d va = _mm256_loadu_pd(a.Data());
        const __m256d vb = _mm256_loadu_pd(b.Data());


        _mm256_storeu_pd(result.Data(), _mm256_mul_pd(va, vb));
    }
    else {
        for (std::size_t i = 0; i < N; ++i) {
            result[i] = static_cast<R>(a[i]) * static_cast<R>(b[i]);
        }
    }

    return result;
}

template <float_num T, float_num U, std::size_t N>
inline auto operator/(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
    ) -> VecSIMDCommon<T, U, N>
{
    using R = std::common_type_t<T, U>;

    VecSIMDCommon<T, U, N> result;

    if constexpr (std::same_as<R, float> && N == 4)
    {
        const __m128 va = _mm_loadu_ps(a.Data());
        const __m128 vb = _mm_loadu_ps(b.Data());

        _mm_storeu_ps(result.Data(), _mm_div_ps(va, vb));
    }
    else if constexpr (std::same_as<R, float> && N == 8)
    {
        const __m256 va = _mm256_loadu_ps(a.Data());
        const __m256 vb = _mm256_loadu_ps(b.Data());

        _mm256_storeu_ps(result.Data(), _mm256_div_ps(va, vb));
    }
    else if constexpr (std::same_as<R, double> && N == 2)
    {
        const __m128d va = _mm_loadu_pd(a.Data());
        const __m128d vb = _mm_loadu_pd(b.Data());

        _mm_storeu_pd(result.Data(), _mm_div_pd(va, vb));
    }
    else if constexpr (std::same_as<R, double> && N == 4)
    {
        const __m256d va = _mm256_loadu_pd(a.Data());
        const __m256d vb = _mm256_loadu_pd(b.Data());

        _mm256_storeu_pd(result.Data(), _mm256_div_pd(va, vb));
    }
    else {
        for (std::size_t i = 0; i < N; ++i) {
            result[i] = static_cast<R>(a[i]) / static_cast<R>(b[i]);
        }
    }

    return result;
}

template <float_num T, float_num U, std::size_t N>
inline auto operator*(
    const VecSIMD<T, N>& v,
    U scalar
    ) -> VecSIMDCommon<T, U, N>
{
    using R = std::common_type_t<T, U>;

    VecSIMDCommon<T, U, N> result;

    if constexpr (std::same_as<R, float> && N == 4)
    {
        const __m128 vv = _mm_loadu_ps(v.Data());
        const __m128 vs = _mm_set_ps1(static_cast<float>(scalar));

        _mm_storeu_ps(result.Data(), _mm_mul_ps(vv, vs));
    }
    else if constexpr (std::same_as<R, float> && N == 8)
    {
        const __m256 vv = _mm256_loadu_ps(v.Data());
        const __m256 vs = _mm256_set1_ps(static_cast<float>(scalar));

        _mm256_storeu_ps(result.Data(), _mm256_mul_ps(vv, vs));
    }
    else if constexpr (std::same_as<R, double> && N == 2)
    {
        const __m128d vv = _mm_loadu_pd(v.Data());
        const __m128d vs = _mm_set1_pd(static_cast<double>(scalar));

        _mm_storeu_pd(result.Data(), _mm_mul_pd(vv, vs));
    }
    else if constexpr (std::same_as<R, double> && N == 4)
    {
        const __m256d vv = _mm256_loadu_pd(v.Data());
        const __m256d vs = _mm256_set1_pd(static_cast<double>(scalar));

        _mm256_storeu_pd(result.Data(), _mm256_mul_pd(vv, vs));
    }
    else {
        for (std::size_t i = 0; i < N; ++i) {
            result[i] = static_cast<R>(v[i]) * static_cast<R>(scalar);
        }
    }

    return result;
}

template <float_num T, float_num U, std::size_t N>
inline auto operator*(
    U scalar,
    const VecSIMD<T, N>& v
    ) -> VecSIMDCommon<T, U, N>
{
    return v * scalar;
}

template <float_num T, float_num U, std::size_t N>
inline auto operator/(
    const VecSIMD<T, N>& v,
    U scalar
    ) -> VecSIMDCommon<T, U, N>
{
    using R = std::common_type_t<T, U>;

    VecSIMDCommon<T, U, N> result;

    if constexpr (std::same_as<R, float> && N == 4)
    {
        const __m128 vv = _mm_loadu_ps(v.Data());
        const __m128 vs = _mm_set1_ps(static_cast<float>(scalar));

        _mm_storeu_ps(result.Data(), _mm_div_ps(vv, vs));
    }
    else if constexpr (std::same_as<R, float> && N == 8)
    {
        const __m256 vv = _mm256_loadu_ps(v.Data());
        const __m256 vs = _mm256_set1_ps(static_cast<float>(scalar));

        _mm256_storeu_ps(result.Data(), _mm256_div_ps(vv, vs));
    }
    else if constexpr (std::same_as<R, double> && N == 2)
    {
        const __m128d vv = _mm_loadu_pd(v.Data());
        const __m128d vs = _mm_set1_pd(static_cast<double>(scalar));

        _mm_storeu_pd(result.Data(), _mm_div_pd(vv, vs));
    }
    else if constexpr (std::same_as<R, double> && N == 4)
    {
        const __m256d vv = _mm256_loadu_pd(v.Data());
        const __m256d vs = _mm256_set1_pd(static_cast<double>(scalar));

        _mm256_storeu_pd(result.Data(), _mm256_div_pd(vv, vs));
    }
    else {
        for (std::size_t i = 0; i < N; ++i) {
            result[i] = static_cast<R>(v[i]) / static_cast<R>(scalar);
        }
    }

    return result;
}

template <float_num T, std::size_t N>
inline constexpr VecSIMD<T, N> VecSIMD<T, N>::operator+() const
{
    return *this;
}

template <float_num T, std::size_t N>
inline constexpr VecSIMD<T, N> VecSIMD<T, N>::operator-() const
{
    VecSIMD result;

    if constexpr (std::same_as<T, float> && N == 4)
    {
        const __m128 zero = _mm_setzero_ps();
        const __m128 value = _mm_loadu_ps(Data());

        _mm_storeu_ps(result.Data(), _mm_sub_ps(zero, value));
    }
    else if constexpr (std::same_as<T, float> && N == 8)
    {
        const __m256 zero = _mm256_setzero_ps();
        const __m256 value = _mm256_loadu_ps(Data());

        _mm256_storeu_ps(result.Data(), _mm256_sub_ps(zero, value));
    }
    else if constexpr (std::same_as<T, double> && N == 2)
    {
        const __m128d zero = _mm_setzero_pd();
        const __m128d value = _mm_loadu_pd(Data());

        _mm_storeu_pd(result.Data(), _mm_sub_pd(zero, value));
    }
    else if constexpr (std::same_as<T, double> && N == 4)
    {
        const __m256d zero = _mm256_setzero_pd();
        const __m256d value = _mm256_loadu_pd(Data());

        _mm256_storeu_pd(result.Data(), _mm256_sub_pd(zero, value));
    }
    else {
        for (std::size_t i = 0; i < N; ++i) {
            result[i] = -m_data[i];
        }
    }

    return result;
}

template <float_num T, float_num U, std::size_t N>
inline auto Dot(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
) -> std::common_type_t<T, U>
{
    using R = std::common_type_t<T, U>;

    if constexpr (std::same_as<R, float>)
    {
        if constexpr (N == 4)
        {
            const __m128 va = _mm_loadu_ps(a.Data());
            const __m128 vb = _mm_loadu_ps(b.Data());
            const __m128 v = _mm_mul_ps(va, vb);

            const __m128 h = _mm_add_ps(v, _mm_movehl_ps(v, v));
            const __m128 s = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));

            return _mm_cvtss_f32(s);
        }
        else if constexpr (N == 8)
        {
            const __m128 va0 = _mm_loadu_ps(a.Data());
            const __m128 vb0 = _mm_loadu_ps(b.Data());
            const __m128 va1 = _mm_loadu_ps(a.Data() + 4);
            const __m128 vb1 = _mm_loadu_ps(b.Data() + 4);

            const __m128 v0 = _mm_mul_ps(va0, vb0);
            const __m128 v1 = _mm_mul_ps(va1, vb1);
            const __m128 v = _mm_add_ps(v0, v1);

            const __m128 h = _mm_add_ps(v, _mm_movehl_ps(v, v));
            const __m128 s = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));

            return _mm_cvtss_f32(s);
        }
    }
    else if constexpr (std::same_as<R, double>)
    {
        if constexpr (N == 2)
        {
            const __m128d va = _mm_loadu_pd(a.Data());
            const __m128d vb = _mm_loadu_pd(b.Data());
            const __m128d v = _mm_mul_pd(va, vb);

            return _mm_cvtsd_f64(
                _mm_add_sd(v, _mm_unpackhi_pd(v, v))
            );
        }
        else if constexpr (N == 4)
        {
            const __m128d va0 = _mm_loadu_pd(a.Data());
            const __m128d vb0 = _mm_loadu_pd(b.Data());
            const __m128d va1 = _mm_loadu_pd(a.Data() + 2);
            const __m128d vb1 = _mm_loadu_pd(b.Data() + 2);

            const __m128d v0 = _mm_mul_pd(va0, vb0);
            const __m128d v1 = _mm_mul_pd(va1, vb1);
            const __m128d v = _mm_add_pd(v0, v1);

            return _mm_cvtsd_f64(
                _mm_add_sd(v, _mm_unpackhi_pd(v, v))
            );
        }
    }

    R result{};
    for (std::size_t i = 0; i < N; ++i)
        result += static_cast<R>(a[i]) * static_cast<R>(b[i]);

    return result;
}

template <float_num T, float_num U>
constexpr auto Cross(
    const VecSIMD<T, 3>& a,
    const VecSIMD<U, 3>& b
) -> VecSIMDCommon<T, U, 3>
{
    using R = std::common_type_t<T, U>;

    if (std::is_constant_evaluated())
    {
        return VecSIMD<R, 3>{
            static_cast<R>(a[1])* static_cast<R>(b[2]) -
                static_cast<R>(a[2]) * static_cast<R>(b[1]),

                static_cast<R>(a[2])* static_cast<R>(b[0]) -
                static_cast<R>(a[0]) * static_cast<R>(b[2]),

                static_cast<R>(a[0])* static_cast<R>(b[1]) -
                static_cast<R>(a[1]) * static_cast<R>(b[0])
        };
    }

    VecSIMD<R, 3> result;

    if constexpr (std::same_as<R, float>)
    {
        const __m128 va = _mm_set_ps(
            0.0f,
            static_cast<float>(a[2]),
            static_cast<float>(a[1]),
            static_cast<float>(a[0])
        );

        const __m128 vb = _mm_set_ps(
            0.0f,
            static_cast<float>(b[2]),
            static_cast<float>(b[1]),
            static_cast<float>(b[0])
        );

        const __m128 ayzx = _mm_shuffle_ps(
            va, va, _MM_SHUFFLE(3, 0, 2, 1)
        );

        const __m128 azxy = _mm_shuffle_ps(
            va, va, _MM_SHUFFLE(3, 1, 0, 2)
        );

        const __m128 byzx = _mm_shuffle_ps(
            vb, vb, _MM_SHUFFLE(3, 0, 2, 1)
        );

        const __m128 bzxy = _mm_shuffle_ps(
            vb, vb, _MM_SHUFFLE(3, 1, 0, 2)
        );

        const __m128 resultSIMD = _mm_sub_ps(
            _mm_mul_ps(ayzx, bzxy),
            _mm_mul_ps(azxy, byzx)
        );

        alignas(16) float values[4];
        _mm_store_ps(values, resultSIMD);

        result[0] = values[0];
        result[1] = values[1];
        result[2] = values[2];
    }
    else
    {
        const __m256d va = _mm256_set_pd(
            0.0,
            static_cast<double>(a[2]),
            static_cast<double>(a[1]),
            static_cast<double>(a[0])
        );

        const __m256d vb = _mm256_set_pd(
            0.0,
            static_cast<double>(b[2]),
            static_cast<double>(b[1]),
            static_cast<double>(b[0])
        );

        const __m256d ayzx = _mm256_permute4x64_pd(va, 0xC9);
        const __m256d azxy = _mm256_permute4x64_pd(va, 0xD2);
        const __m256d byzx = _mm256_permute4x64_pd(vb, 0xC9);
        const __m256d bzxy = _mm256_permute4x64_pd(vb, 0xD2);

        const __m256d resultSIMD = _mm256_sub_pd(
            _mm256_mul_pd(ayzx, bzxy),
            _mm256_mul_pd(azxy, byzx)
        );

        alignas(32) double values[4];
        _mm256_store_pd(values, resultSIMD);

        result[0] = values[0];
        result[1] = values[1];
        result[2] = values[2];
    }

    return result;
}

template <float_num T, float_num U, std::size_t N>
constexpr auto Distance(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
) -> std::common_type_t<T, U>
{
    return (a - b).Length();
}

template <float_num T, float_num U, std::size_t N>
constexpr auto Angle(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
) -> std::common_type_t<T, U>
{
    using R = std::common_type_t<T, U>;

    const R lengthA = a.Length();
    const R lengthB = b.Length();

    if (lengthA <= R{ 0 } || lengthB <= R{ 0 })
        return R{ 0 };

    R cosine = Dot(a, b) / (lengthA * lengthB);

    cosine = std::clamp(cosine, R{ -1 }, R{ 1 });

    return std::acos(cosine);
}

template <float_num T, float_num U, float_num V, std::size_t N>
constexpr auto Lerp(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b,
    V t
) -> VecSIMDCommon<T, U, N>
{
    using R = std::common_type_t<T, U, V>;

    return a + (b - a) * static_cast<R>(t);
}

template <float_num T, float_num U, std::size_t N>
constexpr auto Scale(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
) -> VecSIMDCommon<T, U, N>
{
    return a * b;
}

template <float_num T, float_num U, std::size_t N>
constexpr auto Min(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
) -> VecSIMDCommon<T, U, N>
{
    using R = std::common_type_t<T, U>;

    if (std::is_constant_evaluated())
    {
        VecSIMD<R, N> result;

        for (std::size_t i = 0; i < N; ++i)
            result[i] = std::min(
                static_cast<R>(a[i]),
                static_cast<R>(b[i])
            );

        return result;
    }

    VecSIMD<R, N> result;

    if constexpr (std::same_as<R, float> && N == 4)
    {
        const __m128 va = _mm_loadu_ps(a.Data());
        const __m128 vb = _mm_loadu_ps(b.Data());

        _mm_storeu_ps(result.Data(), _mm_min_ps(va, vb));
    }
    else if constexpr (std::same_as<R, float> && N == 8)
    {
        const __m256 va = _mm256_loadu_ps(a.Data());
        const __m256 vb = _mm256_loadu_ps(b.Data());

        _mm256_storeu_ps(result.Data(), _mm256_min_ps(va, vb));
    }
    else if constexpr (std::same_as<R, double> && N == 2)
    {
        const __m128d va = _mm_loadu_pd(a.Data());
        const __m128d vb = _mm_loadu_pd(b.Data());

        _mm_storeu_pd(result.Data(), _mm_min_pd(va, vb));
    }
    else if constexpr (std::same_as<R, double> && N == 4)
    {
        const __m256d va = _mm256_loadu_pd(a.Data());
        const __m256d vb = _mm256_loadu_pd(b.Data());

        _mm256_storeu_pd(result.Data(), _mm256_min_pd(va, vb));
    }
    else
    {
        for (std::size_t i = 0; i < N; ++i)
            result[i] = std::min(
                static_cast<R>(a[i]),
                static_cast<R>(b[i])
            );
    }

    return result;
}

template <float_num T, float_num U, std::size_t N>
constexpr auto Max(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
) -> VecSIMDCommon<T, U, N>
{
    using R = std::common_type_t<T, U>;

    if (std::is_constant_evaluated())
    {
        VecSIMD<R, N> result;

        for (std::size_t i = 0; i < N; ++i)
            result[i] = std::max(
                static_cast<R>(a[i]),
                static_cast<R>(b[i])
            );

        return result;
    }

    VecSIMD<R, N> result;

    if constexpr (std::same_as<R, float> && N == 4)
    {
        const __m128 va = _mm_loadu_ps(a.Data());
        const __m128 vb = _mm_loadu_ps(b.Data());

        _mm_storeu_ps(result.Data(), _mm_max_ps(va, vb));
    }
    else if constexpr (std::same_as<R, float> && N == 8)
    {
        const __m256 va = _mm256_loadu_ps(a.Data());
        const __m256 vb = _mm256_loadu_ps(b.Data());

        _mm256_storeu_ps(result.Data(), _mm256_max_ps(va, vb));
    }
    else if constexpr (std::same_as<R, double> && N == 2)
    {
        const __m128d va = _mm_loadu_pd(a.Data());
        const __m128d vb = _mm_loadu_pd(b.Data());

        _mm_storeu_pd(result.Data(), _mm_max_pd(va, vb));
    }
    else if constexpr (std::same_as<R, double> && N == 4)
    {
        const __m256d va = _mm256_loadu_pd(a.Data());
        const __m256d vb = _mm256_loadu_pd(b.Data());

        _mm256_storeu_pd(result.Data(), _mm256_max_pd(va, vb));
    }
    else
    {
        for (std::size_t i = 0; i < N; ++i)
            result[i] = std::max(
                static_cast<R>(a[i]),
                static_cast<R>(b[i])
            );
    }

    return result;

}

template <float_num T, float_num U, float_num V, std::size_t N>
constexpr auto MoveTowards(
    const VecSIMD<T, N>& current,
    const VecSIMD<U, N>& target,
    V maxDistanceDelta
) -> VecSIMDCommon<T, U, N>
{
    using R = std::common_type_t<T, U, V>;

    const auto delta = target - current;
    const R distance = delta.Length();
    const R maxDelta = static_cast<R>(maxDistanceDelta);

    if (distance <= maxDelta)
        return target;

    if (distance <= R{ 0 })
        return current;

    return current + delta * (maxDelta / distance);
}

template <float_num T, float_num U, std::size_t N>
constexpr auto Reflect(
    const VecSIMD<T, N>& direction,
    const VecSIMD<U, N>& normal
) -> VecSIMDCommon<T, U, N>
{
    using R = std::common_type_t<T, U>;

    return direction - normal * (R{ 2 } *Dot(direction, normal));
}

template <float_num T>
constexpr VecSIMD<T, 2> Perpendicular(
    const VecSIMD<T, 2>& v
)
{
    return VecSIMD<T, 2>{
        -v[1],
            v[0]
    };
}

template <float_num T, std::size_t N>
inline constexpr T VecSIMD<T, N>::LengthSquared() const
{
    return Dot(*this, *this);
}

template <float_num T, std::size_t N>
inline constexpr T VecSIMD<T, N>::Length() const
{
    return std::sqrt(LengthSquared());
}

template <float_num T, std::size_t N>
inline constexpr VecSIMD<T, N> VecSIMD<T, N>::Normalized() const
{
    const T length = Length();

    if (length <= T{ 0 })
        return *this;

    return *this * (T{ 1 } / length);
}

template <float_num T, std::size_t N>
inline constexpr void VecSIMD<T, N>::Normalize()
{
    *this = Normalized();
}

// idea from: https://stackoverflow.com/a/56766138
// made it simpler and consteval
template <typename T>
static consteval auto TypeName() {
#if defined(__clang__)
    constexpr std::string_view prefix = "[T = ";
    constexpr std::string_view suffix = "]";
    constexpr std::string_view func = __PRETTY_FUNCTION__;
#elif defined(__GNUC__)
    constexpr std::string_view prefix = "[with T = ";
    constexpr std::string_view suffix = "]";
    constexpr std::string_view func = __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
    constexpr std::string_view prefix = "TypeName<";
    constexpr std::string_view suffix = ">(void)";
    constexpr std::string_view func = __FUNCSIG__;
#endif
    constexpr auto start = func.find(prefix) + prefix.size();
    constexpr auto end = func.rfind(suffix);
    return func.substr(start, end - start);
}


template <float_num T, std::size_t N>
struct std::formatter<VecSIMD<T, N>>
{
    std::formatter<T> underlying;

    constexpr auto parse(std::format_parse_context& ctx)
    {
        return underlying.parse(ctx);
    }

    auto format(
        const VecSIMD<T, N>& value,
        std::format_context& ctx
    ) const
    {
        auto out = ctx.out();
        
        out = std::format_to(out, "(");

        for (std::size_t i = 0; i < N; ++i)
        {
            if (i != 0)
                out = std::format_to(out, ", ");

            out = underlying.format(value[i], ctx);
        }

        return std::format_to(out, ")");
    }
};
