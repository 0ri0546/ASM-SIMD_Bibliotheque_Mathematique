#pragma once

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

template <std::floating_point T, std::size_t N>
template <typename... Args>
    requires(sizeof...(Args) == N) &&
(std::convertible_to<Args, T> && ...)
inline constexpr VecSIMD<T, N>::VecSIMD(Args&&... args)
    : m_data{ static_cast<T>(std::forward<Args>(args))... }
{
}

template <std::floating_point T, std::size_t N>
inline constexpr T& VecSIMD<T, N>::operator[](size_type index)
{
    return m_data[index];
}


template <std::floating_point T, std::size_t N>
inline constexpr const T& VecSIMD<T, N>::operator[](size_type index) const
{
    return m_data[index];
}

template <std::floating_point T, std::size_t N>
template <std::size_t I>
inline auto VecSIMD<T, N>::get_m128() const
{
    using traits = simd_traits<T, 128>;

    static_assert((I + 1) * traits::width <= N);

    return traits::load(m_data.data() + I * traits::width);
}

template <std::floating_point T, std::size_t N>
template <std::size_t I>
inline auto VecSIMD<T, N>::get_m256() const
{
    using traits = simd_traits<T, 256>;

    static_assert((I + 1) * traits::width <= N);

    return traits::load(m_data.data() + I * traits::width);
}

template <std::floating_point T, std::size_t N>
inline constexpr T* VecSIMD<T, N>::Data()
{
    return m_data.data();
}

template <std::floating_point T, std::size_t N>
inline constexpr const T* VecSIMD<T, N>::Data() const
{
    return m_data.data();
}

template <std::floating_point T, std::size_t N>
inline constexpr std::size_t VecSIMD<T, N>::Size()
{
    return N;
}

template <std::floating_point T, std::size_t N>
template <std::floating_point U>
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

template <std::floating_point T, std::size_t N>
template <std::floating_point U>
inline constexpr VecSIMD<T, N>& VecSIMD<T, N>::operator+=(
    const VecSIMD<U, N>& other
    )
{
    *this = *this + other;
    return *this;
}

template <std::floating_point T, std::size_t N>
template <std::floating_point U>
inline constexpr VecSIMD<T, N>& VecSIMD<T, N>::operator-=(
    const VecSIMD<U, N>& other
    )
{
    *this = *this - other;
    return *this;
}

template <std::floating_point T, std::size_t N>
template <std::floating_point U>
inline constexpr VecSIMD<T, N>& VecSIMD<T, N>::operator*=(
    const VecSIMD<U, N>& other
    )
{
    *this = *this * other;
    return *this;
}

template <std::floating_point T, std::size_t N>
template <std::floating_point U>
inline constexpr VecSIMD<T, N>& VecSIMD<T, N>::operator/=(
    const VecSIMD<U, N>& other
    )
{
    *this = *this / other;
    return *this;
}

template <std::floating_point T, std::size_t N>
template <std::floating_point U>
inline constexpr VecSIMD<T, N>& VecSIMD<T, N>::operator*=(U scalar)
{
    *this = *this * scalar;
    return *this;
}

template <std::floating_point T, std::size_t N>
template <std::floating_point U>
inline constexpr VecSIMD<T, N>& VecSIMD<T, N>::operator/=(U scalar)
{
    *this = *this / scalar;
    return *this;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
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


    return result;

}

template <std::floating_point T, std::floating_point U, std::size_t N>
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

    return result;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
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

    return result;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
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

    return result;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
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
        const __m128 vs = _mm_set1_ps(static_cast<float>(scalar));

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

    return result;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline auto operator*(
    U scalar,
    const VecSIMD<T, N>& v
    ) -> VecSIMDCommon<T, U, N>
{
    return v * scalar;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
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

    return result;
}

template <std::floating_point T, std::size_t N>
inline constexpr VecSIMD<T, N> VecSIMD<T, N>::operator+() const
{
    return *this;
}

template <std::floating_point T, std::size_t N>
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

    return result;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline auto Dot(
    const VecSIMD<T, N>& a,
    const VecSIMD<U, N>& b
) -> std::common_type_t<T, U>
{
    using R = std::common_type_t<T, U>;

    if constexpr (std::same_as<R, float> && N == 4)
    {
        const __m128 va = _mm_loadu_ps(a.Data());
        const __m128 vb = _mm_loadu_ps(b.Data());
        const __m128 mul = _mm_mul_ps(va, vb);

        alignas(16) float values[4];
        _mm_store_ps(values, mul);

        return values[0] + values[1] + values[2] + values[3];
    }
    else if constexpr (std::same_as<R, float> && N == 8)
    {
        const __m256 va = _mm256_loadu_ps(a.Data());
        const __m256 vb = _mm256_loadu_ps(b.Data());
        const __m256 mul = _mm256_mul_ps(va, vb);

        alignas(32) float values[8];
        _mm256_store_ps(values, mul);

        return values[0] + values[1] + values[2] + values[3]
            + values[4] + values[5] + values[6] + values[7];
    }
    else if constexpr (std::same_as<R, double> && N == 2)
    {
        const __m128d va = _mm_loadu_pd(a.Data());
        const __m128d vb = _mm_loadu_pd(b.Data());
        const __m128d mul = _mm_mul_pd(va, vb);

        alignas(16) double values[2];
        _mm_store_pd(values, mul);

        return values[0] + values[1];
    }
    else
    {
        const __m256d va = _mm256_loadu_pd(a.Data());
        const __m256d vb = _mm256_loadu_pd(b.Data());
        const __m256d mul = _mm256_mul_pd(va, vb);

        alignas(32) double values[4];
        _mm256_store_pd(values, mul);

        return values[0] + values[1] + values[2] + values[3];
    }
}

template <std::floating_point T, std::size_t N>
inline constexpr T VecSIMD<T, N>::LengthSquared() const
{
    return Dot(*this, *this);
}

template <std::floating_point T, std::size_t N>
inline constexpr T VecSIMD<T, N>::Length() const
{
    return static_cast<T>(
        std::sqrt(static_cast<double>(LengthSquared()))
    );
}

template <std::floating_point T, std::size_t N>
inline constexpr VecSIMD<T, N> VecSIMD<T, N>::Normalized() const
{
    const T length = Length();

    if (length <= T{ 0 })
        return *this;

    return *this / length;
}

template <std::floating_point T, std::size_t N>
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


template <std::floating_point T, std::size_t N>
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

        out = std::format_to(out, "{}", TypeName<VecSIMD<T, N>>());
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
