#pragma once

template <>
struct simd_traits<float, 128> {
    using m128 = __m128;
    static constexpr std::size_t width = 4;

    static m128 load(const float* ptr) {
        return _mm_loadu_ps(ptr);
    }
};

template <>
struct simd_traits<double, 128> {
    using m128 = __m128d;
    static constexpr std::size_t width = 2;

    static m128 load(const double* ptr) {
        return _mm_loadu_pd(ptr);
    }
};

template <>
struct simd_traits<float, 256> {
    using m256 = __m256;
    static constexpr std::size_t width = 8;

    static m256 load(const float* ptr) {
        return _mm256_loadu_ps(ptr);
    }
};

template <>
struct simd_traits<double, 256> {
    using m256 = __m256d;
    static constexpr std::size_t width = 4;

    static m256 load(const double* ptr) {
        return _mm256_loadu_pd(ptr);
    }
};

template <std::floating_point T, std::size_t N>
template <typename... Args>
    requires(sizeof...(Args) == N) && (std::convertible_to<Args, T> && ...)
inline constexpr Vec<T, N>::Vec(Args&&... args) : m_data{ static_cast<T>(std::forward<Args>(args))... } {}

template <std::floating_point T, std::size_t N>
inline constexpr T& Vec<T, N>::operator[](size_type index) {
    return m_data[index];
}

template <std::floating_point T, std::size_t N>
inline constexpr const T& Vec<T, N>::operator[](size_type index) const {
    return m_data[index];
}

template <std::floating_point T, std::size_t N>
template <std::size_t I>
inline auto Vec<T, N>::get_m128() const
{
    using traits = simd_traits<T, 128>;

    static_assert(I * traits::width < N);

    return traits::load(m_data.data() + I * traits::width);
}

template <std::floating_point T, std::size_t N>
template <std::size_t I>
inline auto Vec<T, N>::get_m256() const
{
    using traits = simd_traits<T, 256>;

    static_assert(I * traits::width < N);

    return traits::load(m_data.data() + I * traits::width);
}

template <std::floating_point T, std::size_t N>
inline constexpr T* Vec<T, N>::Data() {
    return m_data.data();
}

template <std::floating_point T, std::size_t N>
inline constexpr const T* Vec<T, N>::Data() const {
    return m_data.data();
}

template <std::floating_point T, std::size_t N>
inline constexpr std::size_t Vec<T, N>::Size() {
    return N;
}

template <std::floating_point T, std::size_t N>
template <std::floating_point U>
inline constexpr bool Vec<T, N>::operator==(const Vec<U, N>& other) const {
    for (size_type i = 0; i < N; ++i) {
        if ((*this)[i] != other[i]) {
            return false;
        }
    }

    return true;
}

template <std::floating_point T, std::size_t N>
template <std::floating_point U>
inline constexpr Vec<T, N>& Vec<T, N>::operator+=(const Vec<U, N>& other) {
    for (size_type i = 0; i < N; ++i) {
        (*this)[i] += other[i];
    }

    return *this;
}

template <std::floating_point T, std::size_t N>
template <std::floating_point U>
inline constexpr Vec<T, N>& Vec<T, N>::operator-=(const Vec<U, N>& other) {
    for (size_type i = 0; i < N; ++i) {
        (*this)[i] -= other[i];
    }

    return *this;
}

template <std::floating_point T, std::size_t N>
template <std::floating_point U>
inline constexpr Vec<T, N>& Vec<T, N>::operator*=(const Vec<U, N>& other) {
    for (size_type i = 0; i < N; ++i) {
        (*this)[i] *= other[i];
    }

    return *this;
}

template <std::floating_point T, std::size_t N>
template <std::floating_point U>
inline constexpr Vec<T, N>& Vec<T, N>::operator/=(const Vec<U, N>& other) {
    for (size_type i = 0; i < N; ++i) {
        (*this)[i] /= other[i];
    }

    return *this;
}

template <std::floating_point T, std::size_t N>
template <std::floating_point U>
inline constexpr Vec<T, N>& Vec<T, N>::operator*=(U scalar) {
    for (auto& value : m_data) {
        value *= scalar;
    }

    return *this;
}

template <std::floating_point T, std::size_t N>
template <std::floating_point U>
inline constexpr Vec<T, N>& Vec<T, N>::operator/=(U scalar) {
    for (auto& value : m_data) {
        value /= scalar;
    }

    return *this;
}

template <std::floating_point T, std::size_t N>
inline constexpr Vec<T, N> Vec<T, N>::operator+() const {
    return *this;
}

template <std::floating_point T, std::size_t N>
inline constexpr Vec<T, N> Vec<T, N>::operator-() const {
    Vec result;

    for (size_type i = 0; i < N; ++i) {
        result[i] = -(*this)[i];
    }

    return result;
}

template <std::floating_point T, std::size_t N>
inline constexpr T Vec<T, N>::LengthSquared() const {
    T result{};

    for (size_type i = 0; i < N; ++i) {
        result += (*this)[i] * (*this)[i];
    }

    return result;
}

template <std::floating_point T, std::size_t N>
inline constexpr T Vec<T, N>::Length() const {
    return static_cast<T>(std::sqrt(static_cast<double>(LengthSquared())));
}

template <std::floating_point T, std::size_t N>
inline constexpr Vec<T, N> Vec<T, N>::Normalized() const {
    const T length = Length();

    if (length <= T{ 0 }) {
        return *this;
    }

    return *this * (T{ 1 } / length);
}

template <std::floating_point T, std::size_t N>
inline constexpr void Vec<T, N>::Normalize() {
    *this = Normalized();
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto operator+(const Vec<T, N>& a, const Vec<U, N>& b)
-> VecCommon<T, U, N>
{
    using R = std::common_type_t<T, U>;
    VecCommon<T, U, N> result;

    if constexpr (std::same_as<R, float> && N == 4)
    {
        const __m128 va = _mm_loadu_ps(a.Data());
        const __m128 vb = _mm_loadu_ps(b.Data());

        _mm_storeu_ps(
            result.Data(),
            _mm_add_ps(va, vb)
        );
    }
    else if constexpr (std::same_as<R, float> && N == 8)
    {
        const __m256 va = _mm256_loadu_ps(a.Data());
        const __m256 vb = _mm256_loadu_ps(b.Data());

        _mm256_storeu_ps(
            result.Data(),
            _mm256_add_ps(va, vb)
        );
    }
    else if constexpr (std::same_as<R, double> && N == 2)
    {
        const __m128d va = _mm_loadu_pd(a.Data());
        const __m128d vb = _mm_loadu_pd(b.Data());

        _mm_storeu_pd(
            result.Data(),
            _mm_add_pd(va, vb)
        );
    }
    else if constexpr (std::same_as<R, double> && N == 4)
    {
        const __m256d va = _mm256_loadu_pd(a.Data());
        const __m256d vb = _mm256_loadu_pd(b.Data());

        _mm256_storeu_pd(
            result.Data(),
            _mm256_add_pd(va, vb)
        );
    }

    return result;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto operator-(const Vec<T, N>& a, const Vec<U, N>& b) -> VecCommon<T, U, N> {
    using R = std::common_type_t<T, U>;

    VecCommon<T, U, N> result;

    for (std::size_t i = 0; i < N; ++i) {
        result[i] = static_cast<R>(a[i]) - static_cast<R>(b[i]);
    }

    return result;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto operator*(const Vec<T, N>& a, const Vec<U, N>& b) -> VecCommon<T, U, N> {
    using R = std::common_type_t<T, U>;

    VecCommon<T, U, N> result;

    for (std::size_t i = 0; i < N; ++i) {
        result[i] = static_cast<R>(a[i]) * static_cast<R>(b[i]);
    }

    return result;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto operator/(const Vec<T, N>& a, const Vec<U, N>& b) -> VecCommon<T, U, N> {
    using R = std::common_type_t<T, U>;

    VecCommon<T, U, N> result;

    for (std::size_t i = 0; i < N; ++i) {
        result[i] = static_cast<R>(a[i]) / static_cast<R>(b[i]);
    }

    return result;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto operator*(const Vec<T, N>& v, U scalar) -> VecCommon<T, U, N> {
    using R = std::common_type_t<T, U>;

    VecCommon<T, U, N> result;

    for (std::size_t i = 0; i < N; ++i) {
        result[i] = static_cast<R>(v[i]) * static_cast<R>(scalar);
    }

    return result;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto operator*(U scalar, const Vec<T, N>& v) -> VecCommon<T, U, N> {
    return v * scalar;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto operator/(const Vec<T, N>& v, U scalar) -> VecCommon<T, U, N> {
    using R = std::common_type_t<T, U>;

    VecCommon<T, U, N> result;

    for (std::size_t i = 0; i < N; ++i) {
        result[i] = static_cast<R>(v[i]) / static_cast<R>(scalar);
    }

    return result;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto Dot(const Vec<T, N>& a, const Vec<U, N>& b) -> std::common_type_t<T, U> {
    using R = std::common_type_t<T, U>;

    R result{};

    for (std::size_t i = 0; i < N; ++i) {
        result += static_cast<R>(a[i]) * static_cast<R>(b[i]);
    }

    return result;
}

template <std::floating_point T, std::floating_point U>
inline constexpr auto Cross(const Vec<T, 3>& a, const Vec<U, 3>& b) -> VecCommon<T, U, 3> {
    using R = std::common_type_t<T, U>;

    return VecCommon<T, U, 3>{
        static_cast<R>(a[1])* static_cast<R>(b[2]) - static_cast<R>(a[2]) * static_cast<R>(b[1]),
            static_cast<R>(a[2])* static_cast<R>(b[0]) - static_cast<R>(a[0]) * static_cast<R>(b[2]),
            static_cast<R>(a[0])* static_cast<R>(b[1]) - static_cast<R>(a[1]) * static_cast<R>(b[0])};
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto Distance(const Vec<T, N>& a, const Vec<U, N>& b) -> std::common_type_t<T, U> {
    return (a - b).Length();
}

template <std::floating_point T, std::floating_point U, std::floating_point V, std::size_t N>
inline constexpr auto Lerp(const Vec<T, N>& a, const Vec<U, N>& b, V t) -> VecCommon<T, U, N> {
    using R = std::common_type_t<T, U>;

    VecCommon<T, U, N> result;

    for (std::size_t i = 0; i < N; ++i) {
        const R av = static_cast<R>(a[i]);
        const R bv = static_cast<R>(b[i]);
        result[i] = av + (bv - av) * static_cast<R>(t);
    }

    return result;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto Scale(const Vec<T, N>& a, const Vec<U, N>& b) -> VecCommon<T, U, N> {
    return a * b;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto Min(const Vec<T, N>& a, const Vec<U, N>& b) -> VecCommon<T, U, N> {
    using R = std::common_type_t<T, U>;

    VecCommon<T, U, N> result;

    for (std::size_t i = 0; i < N; ++i) {
        const R av = static_cast<R>(a[i]);
        const R bv = static_cast<R>(b[i]);
        result[i] = av < bv ? av : bv;
    }

    return result;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto Max(const Vec<T, N>& a, const Vec<U, N>& b) -> VecCommon<T, U, N> {
    using R = std::common_type_t<T, U>;

    VecCommon<T, U, N> result;

    for (std::size_t i = 0; i < N; ++i) {
        const R av = static_cast<R>(a[i]);
        const R bv = static_cast<R>(b[i]);
        result[i] = av > bv ? av : bv;
    }

    return result;
}

template <std::floating_point T, std::floating_point U, std::floating_point V, std::size_t N>
inline constexpr auto MoveTowards(const Vec<T, N>& current, const Vec<U, N>& target, V maxDistanceDelta)
-> VecCommon<T, U, N> {
    using R = std::common_type_t<T, U>;

    const VecCommon<T, U, N> delta = target - current;
    const R distance = delta.Length();

    if (distance <= static_cast<R>(maxDistanceDelta) || distance <= R{ 0 }) {
        VecCommon<T, U, N> result;

        for (std::size_t i = 0; i < N; ++i) {
            result[i] = static_cast<R>(target[i]);
        }

        return result;
    }

    VecCommon<T, U, N> result;

    for (std::size_t i = 0; i < N; ++i) {
        result[i] = static_cast<R>(current[i]) + delta[i] * (static_cast<R>(maxDistanceDelta) / distance);
    }

    return result;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto Reflect(const Vec<T, N>& v, const Vec<U, N>& normal) -> VecCommon<T, U, N> {
    using R = std::common_type_t<T, U>;

    VecCommon<T, U, N> result;
    const R factor = R{ 2 } *Dot(v, normal);

    for (std::size_t i = 0; i < N; ++i) {
        result[i] = static_cast<R>(v[i]) - static_cast<R>(normal[i]) * factor;
    }

    return result;
}

template <std::floating_point T, std::floating_point U, std::size_t N>
inline constexpr auto Angle(const Vec<T, N>& a, const Vec<U, N>& b) -> std::common_type_t<T, U> {
    using R = std::common_type_t<T, U>;

    R lengthSquaredA{};
    R lengthSquaredB{};

    for (std::size_t i = 0; i < N; ++i) {
        lengthSquaredA += static_cast<R>(a[i]) * static_cast<R>(a[i]);
        lengthSquaredB += static_cast<R>(b[i]) * static_cast<R>(b[i]);
    }

    const R denom = static_cast<R>(std::sqrt(static_cast<double>(lengthSquaredA))) *
        static_cast<R>(std::sqrt(static_cast<double>(lengthSquaredB)));

    if (denom <= R{ 0 }) {
        return R{ 0 };
    }

    R cosAngle = Dot(a, b) / denom;
    cosAngle = cosAngle < R{ -1 } ? R{ -1 } : (cosAngle > R{ 1 } ? R{ 1 } : cosAngle);

    return static_cast<R>(std::acos(static_cast<double>(cosAngle)));
}

template <std::floating_point T>
inline constexpr Vec<T, 2> Perpendicular(const Vec<T, 2>& v) {
    return Vec<T, 2>{-v[1], v[0]};
}

template <>
struct std::formatter<__m128> {

    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const __m128& obj, std::format_context& ctx) const {
        auto out = ctx.out();
        out = std::format_to(out, "(");

        for (std::size_t i = 0; i < 4; ++i) {
            if (i != 0) {
                out = std::format_to(out, ", ");
            }
            out = std::format_to(out, "{}", obj.m128_f32[i]);
        }

        return std::format_to(out, ")");
    }
};

template <>
struct std::formatter<__m128d> {

    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const __m128d& obj, std::format_context& ctx) const {
        auto out = ctx.out();
        out = std::format_to(out, "(");

        for (std::size_t i = 0; i < 2; ++i) {
            if (i != 0) {
                out = std::format_to(out, ", ");
            }
            out = std::format_to(out, "{}", obj.m128d_f64[i]);
        }

        return std::format_to(out, ")");
    }
};

template <>
struct std::formatter<__m256> {

    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const __m256& obj, std::format_context& ctx) const {
        auto out = ctx.out();
        out = std::format_to(out, "(");

        for (std::size_t i = 0; i < 8; ++i) {
            if (i != 0) {
                out = std::format_to(out, ", ");
            }
            out = std::format_to(out, "{}", obj.m256_f32[i]);
        }

        return std::format_to(out, ")");
    }
};

template <>
struct std::formatter<__m256d> {

    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const __m256d& obj, std::format_context& ctx) const {
        auto out = ctx.out();
        out = std::format_to(out, "(");

        for (std::size_t i = 0; i < 4; ++i) {
            if (i != 0) {
                out = std::format_to(out, ", ");
            }
            out = std::format_to(out, "{}", obj.m256d_f64[i]);
        }

        return std::format_to(out, ")");
    }
};

template <std::floating_point T>
struct std::formatter<simd_traits<T, 128>> {

    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const simd_traits<T, 128>& obj, std::format_context& ctx) const {
        auto out = ctx.out();
        out = std::format_to(out, "{}", obj.m128);
    }
};

template <std::floating_point T>
struct std::formatter<simd_traits<T, 256>> {

    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const simd_traits<T, 256>& obj, std::format_context& ctx) const {
        auto out = ctx.out();
        out = std::format_to(out, "{}", obj.m256);
    }
};

template <std::floating_point T, std::size_t N>
struct std::formatter<Vec<T, N>> {
    std::formatter<T> element_formatter;

    constexpr auto parse(std::format_parse_context& ctx) {
        return element_formatter.parse(ctx);
    }

    auto format(const Vec<T, N>& obj, std::format_context& ctx) const {
        auto out = ctx.out();
        out = std::format_to(out, "(");

        for (std::size_t i = 0; i < N; ++i) {
            if (i != 0) {
                out = std::format_to(out, ", ");
            }
            out = std::format_to(out, "{}", obj[i]);
        }

        return std::format_to(out, ")");
    }
};
