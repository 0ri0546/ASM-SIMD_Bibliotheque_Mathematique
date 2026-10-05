#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <print>
#include <type_traits>
#include <utility>

// Batch helpers (AVX2 at most)
namespace Detail {

    // Calls f.template operator()<traits>(i) for every batch of [0, count):
    // 256-bit batches first, then 128-bit batches, then a scalar tail (traits width 0 = one element).
    template <float_num T, typename F>
    inline constexpr void ForEachBatch(std::size_t count, F&& f) {
        using wide = simd_traits<T, 256>;
        using narrow = simd_traits<T, 128>;
        using scalar = simd_traits<T, 0>;

        std::size_t i = 0;

        for (; i + wide::width <= count; i += wide::width) {
            f.template operator()<wide>(i);
        }

        for (; i + narrow::width <= count; i += narrow::width) {
            f.template operator()<narrow>(i);
        }

        for (; i < count; i += scalar::width) {
            f.template operator()<scalar>(i);
        }
    }

    // Same batching, but f returns bool and the loop stops at the first false.
    template <float_num T, typename F>
    inline constexpr bool AllOfBatches(std::size_t count, F&& f) {
        using wide = simd_traits<T, 256>;
        using narrow = simd_traits<T, 128>;
        using scalar = simd_traits<T, 0>;

        std::size_t i = 0;

        for (; i + wide::width <= count; i += wide::width) {
            if (!f.template operator()<wide>(i)) {
                return false;
            }
        }

        for (; i + narrow::width <= count; i += narrow::width) {
            if (!f.template operator()<narrow>(i)) {
                return false;
            }
        }

        for (; i < count; i += scalar::width) {
            if (!f.template operator()<scalar>(i)) {
                return false;
            }
        }

        return true;
    }

} // namespace Detail

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <typename... Args>
    requires(sizeof...(Args) == RowCount * ColCount) && (std::convertible_to<Args, T> && ...)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>::MatSIMD(Args&&... args) {
    constexpr std::size_t total_elements = RowCount * ColCount;

    // The arguments are given row by row: values[row * ColCount + col]
    const std::array<T, total_elements> values{ static_cast<T>(std::forward<Args>(args))... };

    if constexpr (std::same_as<Layout, RowMajor>) {
        // Same order as m_data: copy straight over
        Detail::ForEachBatch<T>(total_elements, [&]<typename traits>(size_type i) {
            traits::store(m_data.data() + i, traits::load(values.data() + i));
        });
    }
    else {
        // Other order: slot k of m_data takes values[source_index[k]]
        constexpr auto source_index = []() {
            std::array<std::int32_t, total_elements> table{};

            for (std::int32_t row = 0; row < RowCount; ++row) {
                for (std::int32_t col = 0; col < ColCount; ++col) {
                    table[m_mapping(row, col)] = static_cast<std::int32_t>(row * ColCount + col);
                }
            }

            return table;
        }();

        Detail::ForEachBatch<T>(total_elements, [&]<typename traits>(std::int32_t i) {
            traits::store(m_data.data() + i, traits::gather(values.data(), source_index.data() + i));
        });
    }
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, typename OtherLayout>
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>::MatSIMD(const MatSIMD<U, RowCount, ColCount, OtherLayout>& other) {
    constexpr std::size_t total_elements = RowCount * ColCount;

    if constexpr (std::same_as<T, U> && std::same_as<Layout, OtherLayout>) {
        Detail::ForEachBatch<T>(total_elements, [&]<typename traits>(size_type i) {
            traits::store(m_data.data() + i, traits::load(other.Data() + i));
        });
    }
    else if constexpr (std::same_as<T&, U>) {
        // Same type, other layout: reorder the elements first
        *this = MatCastLayout<T, RowCount, ColCount, OtherLayout, Layout>(other);
    }
    else {
        //std::println("SIMD convert fallback");
        for (size_type row = 0; row < RowCount; ++row) {
            for (size_type col = 0; col < ColCount; ++col) {
                (*this)[row, col] = static_cast<T>(other[row, col]);
            }
        }
    }
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr T& MatSIMD<T, RowCount, ColCount, Layout>::operator[](size_type row, size_type col) {
    return m_data[m_mapping(row, col)];
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr const T& MatSIMD<T, RowCount, ColCount, Layout>::operator[](size_type row, size_type col) const {
    return m_data[m_mapping(row, col)];
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr T& MatSIMD<T, RowCount, ColCount, Layout>::operator()(size_type row, size_type col) {
    return (*this)[row, col];
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr const T& MatSIMD<T, RowCount, ColCount, Layout>::operator()(size_type row, size_type col) const {
    return (*this)[row, col];
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr T* MatSIMD<T, RowCount, ColCount, Layout>::Data() {
    return m_data.data();
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr const T* MatSIMD<T, RowCount, ColCount, Layout>::Data() const {
    return m_data.data();
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto MatSIMD<T, RowCount, ColCount, Layout>::MDSpan() {
    return mdspan_type{ m_data.data() };
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto MatSIMD<T, RowCount, ColCount, Layout>::MDSpan() const {
    return const_mdspan_type{ m_data.data() };
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr std::size_t MatSIMD<T, RowCount, ColCount, Layout>::Rows() {
    return RowCount;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr std::size_t MatSIMD<T, RowCount, ColCount, Layout>::Cols() {
    return ColCount;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr std::size_t MatSIMD<T, RowCount, ColCount, Layout>::Size() {
    return RowCount * ColCount;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U>
inline constexpr bool
MatSIMD<T, RowCount, ColCount, Layout>::operator==(const MatSIMD<U, RowCount, ColCount, Layout>& other) const {
    constexpr std::size_t total_elements = RowCount * ColCount;

    if constexpr (std::same_as<T, U>) {
        return Detail::AllOfBatches<T>(total_elements, [&]<typename traits>(size_type i) {
            auto data_vec = traits::load(m_data.data() + i);
            auto other_vec = traits::load(other.Data() + i);

            return traits::all_equal(data_vec, other_vec);
        });
    }
    else {
        std::println("SIMD equality fallback");
        for (size_type row = 0; row < RowCount; ++row) {
            for (size_type col = 0; col < ColCount; ++col) {
                if ((*this)[row, col] != other[row, col]) {
                    return false;
                }
            }
        }

        return true;
    }
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, typename OtherLayout>
inline constexpr bool
MatSIMD<T, RowCount, ColCount, Layout>::operator==(const MatSIMD<U, RowCount, ColCount, OtherLayout>& other) const {
    // Elements are stored in a different order: reorder `other` to our layout, then use the same-layout ==
    return *this == MatCastLayout<U, RowCount, ColCount, OtherLayout, Layout>(other);
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, typename OtherLayout>
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>&
MatSIMD<T, RowCount, ColCount, Layout>::operator+=(const MatSIMD<U, RowCount, ColCount, OtherLayout>& other) {
    constexpr std::size_t total_elements = RowCount * ColCount;

    if constexpr (!std::same_as<Layout, OtherLayout>) {
        // Elements are stored in a different order: reorder `other` to our layout first
        return *this += MatCastLayout<U, RowCount, ColCount, OtherLayout, Layout>(other);
    }
    else if constexpr (std::same_as<T, U>) {
        Detail::ForEachBatch<T>(total_elements, [&]<typename traits>(size_type i) {
            auto data_vec = traits::load(m_data.data() + i);
            auto other_vec = traits::load(other.Data() + i);
            auto res_vec = traits::add(data_vec, other_vec);

            traits::store(m_data.data() + i, res_vec);
        });
    }
    else {
        std::println("SIMD add fallback");
        for (size_type row = 0; row < RowCount; ++row) {
            for (size_type col = 0; col < ColCount; ++col) {
                (*this)[row, col] += other[row, col];
            }
        }
    }

    return *this;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, typename OtherLayout>
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>&
MatSIMD<T, RowCount, ColCount, Layout>::operator-=(const MatSIMD<U, RowCount, ColCount, OtherLayout>& other) {
    constexpr std::size_t total_elements = RowCount * ColCount;

    if constexpr (!std::same_as<Layout, OtherLayout>) {
        // Elements are stored in a different order: reorder `other` to our layout first
        return *this -= MatCastLayout<U, RowCount, ColCount, OtherLayout, Layout>(other);
    }
    else if constexpr (std::same_as<T, U>) {
        Detail::ForEachBatch<T>(total_elements, [&]<typename traits>(size_type i) {
            auto data_vec = traits::load(m_data.data() + i);
            auto other_vec = traits::load(other.Data() + i);
            auto res_vec = traits::sub(data_vec, other_vec);

            traits::store(m_data.data() + i, res_vec);
        });
    }
    else {
        std::println("SIMD sub fallback");
        for (size_type row = 0; row < RowCount; ++row) {
            for (size_type col = 0; col < ColCount; ++col) {
                (*this)[row, col] -= other[row, col];
            }
        }
    }

    return *this;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U>
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>& MatSIMD<T, RowCount, ColCount, Layout>::operator*=(U scalar) {
    constexpr std::size_t total_elements = RowCount * ColCount;

    if constexpr (std::assignable_from<T&, U>) {
        Detail::ForEachBatch<T>(total_elements, [&]<typename traits>(size_type i) {
            auto data_vec = traits::load(m_data.data() + i);
            auto scalar_vec = traits::set1(static_cast<T>(scalar));
            auto res_vec = traits::mul(data_vec, scalar_vec);

            traits::store(m_data.data() + i, res_vec);
        });
    }
    else {
        std::println("SIMD mul fallback");
        for (auto& value : m_data) {
            value *= scalar;
        }
    }

    return *this;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U>
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>& MatSIMD<T, RowCount, ColCount, Layout>::operator/=(U scalar) {
    constexpr std::size_t total_elements = RowCount * ColCount;

    if constexpr (std::assignable_from<T&, U>) {
        Detail::ForEachBatch<T>(total_elements, [&]<typename traits>(size_type i) {
            auto data_vec = traits::load(m_data.data() + i);
            auto scalar_vec = traits::set1(static_cast<T>(scalar));
            auto res_vec = traits::div(data_vec, scalar_vec);

            traits::store(m_data.data() + i, res_vec);
        });
    }
    else {
        std::println("SIMD div fallback");
        for (auto& value : m_data) {
            value /= scalar;
        }
    }

    return *this;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::operator+() const {
    return *this;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::operator-() const {
    MatSIMD result;

    constexpr std::size_t total_elements = RowCount * ColCount;

    Detail::ForEachBatch<T>(total_elements, [&]<typename traits>(size_type i) {
        auto data_vec = traits::load(m_data.data() + i);
        auto res_vec = traits::neg(data_vec);

        traits::store(result.m_data.data() + i, res_vec);
    });

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr MatSIMD<T, ColCount, RowCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::Transpose() const {
    using result_type = MatSIMD<T, ColCount, RowCount, Layout>;

    constexpr std::size_t total_elements = RowCount * ColCount;

    result_type result;

    // Slot k of the result takes m_data[source_index[k]]: result(col, row) = (*this)(row, col)
    constexpr auto source_index = []() {
        typename result_type::mapping_type result_mapping{};
        std::array<std::int32_t, total_elements> table{};

        for (size_type row = 0; row < RowCount; ++row) {
            for (size_type col = 0; col < ColCount; ++col) {
                table[result_mapping(col, row)] = static_cast<std::int32_t>(m_mapping(row, col));
            }
        }

        return table;
        }();

    Detail::ForEachBatch<T>(total_elements, [&]<typename traits>(size_type i) {
        traits::store(result.Data() + i, traits::gather(m_data.data(), source_index.data() + i));
    });

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <std::size_t N>
    requires(RowCount == ColCount && N == RowCount)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::Identity() {
    MatSIMD result;

    // m_data is zero-initialised: only the diagonal has to be written
    for (size_type i = 0; i < RowCount; ++i) {
        result[i, i] = T{ 1 };
    }

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <std::size_t N>
    requires(RowCount == ColCount && N == RowCount)
inline constexpr auto MatSIMD<T, RowCount, ColCount, Layout>::Determinant() const {
    static_assert(RowCount == ColCount);

    // Row-major working copy: each row is a contiguous run of ColCount elements
    std::array<T, RowCount* ColCount> temp{};

    for (size_type row = 0; row < RowCount; ++row) {
        for (size_type col = 0; col < ColCount; ++col) {
            temp[row * ColCount + col] = (*this)[row, col];
        }
    }

    T result = T{ 1 };

    for (size_type i = 0; i < RowCount; ++i) {
        const T pivot = temp[i * ColCount + i];

        if (pivot == T{}) {
            return T{};
        }

        result *= pivot;

        const T* pivot_row = temp.data() + i * ColCount;

        for (size_type row = i + 1; row < RowCount; ++row) {
            const T factor = temp[row * ColCount + i] / pivot;
            T* target_row = temp.data() + row * ColCount;

            // target_row[col] -= factor * pivot_row[col], for col in [i, ColCount)
            Detail::ForEachBatch<T>(ColCount - i, [&]<typename traits>(size_type k) {
                auto row_vec = traits::load(target_row + i + k);
                auto pivot_vec = traits::load(pivot_row + i + k);
                auto factor_vec = traits::set1(factor);
                auto res_vec = traits::sub(row_vec, traits::mul(factor_vec, pivot_vec));

                traits::store(target_row + i + k, res_vec);
            });
        }
    }

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <std::size_t N>
    requires(RowCount == ColCount && N == RowCount)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::Inverse() const {
    static_assert(RowCount == ColCount);

    constexpr std::size_t augmented_cols = ColCount * 2;

    // Row-major [A | I]: each row is a contiguous run of augmented_cols elements
    std::array<T, RowCount* augmented_cols> augmented{};

    for (size_type row = 0; row < RowCount; ++row) {
        for (size_type col = 0; col < ColCount; ++col) {
            augmented[row * augmented_cols + col] = (*this)[row, col];
        }

        augmented[row * augmented_cols + ColCount + row] = T{ 1 };
    }

    for (size_type i = 0; i < RowCount; ++i) {
        const T pivot = augmented[i * augmented_cols + i];

        if (pivot == T{}) {
            return MatSIMD{};
        }

        T* pivot_row = augmented.data() + i * augmented_cols;

        // pivot_row[col] /= pivot
        Detail::ForEachBatch<T>(augmented_cols, [&]<typename traits>(size_type k) {
            auto row_vec = traits::load(pivot_row + k);
            auto pivot_vec = traits::set1(pivot);

            traits::store(pivot_row + k, traits::div(row_vec, pivot_vec));
        });

        for (size_type row = 0; row < RowCount; ++row) {
            if (row == i) {
                continue;
            }

            const T factor = augmented[row * augmented_cols + i];
            T* target_row = augmented.data() + row * augmented_cols;

            // target_row[col] -= factor * pivot_row[col]
            Detail::ForEachBatch<T>(augmented_cols, [&]<typename traits>(size_type k) {
                auto row_vec = traits::load(target_row + k);
                auto pivot_vec = traits::load(pivot_row + k);
                auto factor_vec = traits::set1(factor);
                auto res_vec = traits::sub(row_vec, traits::mul(factor_vec, pivot_vec));

                traits::store(target_row + k, res_vec);
            });
        }
    }

    // The right half of [A | I] is now the inverse
    MatSIMD result;

    for (size_type row = 0; row < RowCount; ++row) {
        for (size_type col = 0; col < ColCount; ++col) {
            result[row, col] = augmented[row * augmented_cols + ColCount + col];
        }
    }

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::Zero() {
    return MatSIMD{};
}

// The 4x4 builders below give their 16 values (row by row) to the variadic constructor,
// which stores them with SIMD (gather for column-major).

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::RotationX(U angleRadians) {
    const U s = std::sin(angleRadians);
    const U c = std::cos(angleRadians);

    return MatSIMD{
        T{1}, T{0},             T{0},              T{0},
        T{0}, static_cast<T>(c), static_cast<T>(-s), T{0},
        T{0}, static_cast<T>(s), static_cast<T>(c),  T{0},
        T{0}, T{0},             T{0},              T{1} };
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::RotationY(U angleRadians) {
    const U s = std::sin(angleRadians);
    const U c = std::cos(angleRadians);

    return MatSIMD{
        static_cast<T>(c),  T{0}, static_cast<T>(s), T{0},
        T{0},              T{1}, T{0},             T{0},
        static_cast<T>(-s), T{0}, static_cast<T>(c), T{0},
        T{0},              T{0}, T{0},             T{1} };
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::RotationZ(U angleRadians) {
    const U s = std::sin(angleRadians);
    const U c = std::cos(angleRadians);

    return MatSIMD{
        static_cast<T>(c), static_cast<T>(-s), T{0}, T{0},
        static_cast<T>(s), static_cast<T>(c),  T{0}, T{0},
        T{0},             T{0},              T{1}, T{0},
        T{0},             T{0},              T{0}, T{1} };
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::Scale(const VecSIMD<U, 3>& scale) {
    return MatSIMD{
        static_cast<T>(scale[0]), T{0},                    T{0},                    T{0},
        T{0},                    static_cast<T>(scale[1]), T{0},                    T{0},
        T{0},                    T{0},                    static_cast<T>(scale[2]), T{0},
        T{0},                    T{0},                    T{0},                    T{1} };
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>
MatSIMD<T, RowCount, ColCount, Layout>::Translate(const VecSIMD<U, 3>& translation) {
    return MatSIMD{
        T{1}, T{0}, T{0}, static_cast<T>(translation[0]),
        T{0}, T{1}, T{0}, static_cast<T>(translation[1]),
        T{0}, T{0}, T{1}, static_cast<T>(translation[2]),
        T{0}, T{0}, T{0}, T{1} };
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>
MatSIMD<T, RowCount, ColCount, Layout>::Rotate(const QuaternionSIMD<U>& rotation) {
    // One register = 4 lanes: SSE for float, AVX for double
    using traits = std::conditional_t<std::same_as<T, float>, simd_traits<T, 128>, simd_traits<T, 256>>;

    // We build the 4 rows of a matrix and store them one after the other. Row-major memory is the
    // matrix itself. Column-major memory is its transpose, and the transpose of a rotation matrix
    // is the rotation of the conjugate quaternion (-x, -y, -z, w): flip the vector part.
    constexpr T sign = std::same_as<Layout, ColumnMajor> ? T{ -1 } : T{ 1 };

    const T x = sign * static_cast<T>(rotation.X());
    const T y = sign * static_cast<T>(rotation.Y());
    const T z = sign * static_cast<T>(rotation.Z());
    const T w = static_cast<T>(rotation.W());

    const auto zero = traits::set1(T{ 0 });

    const auto q = traits::set4(x, y, z, w);   // (x,   y,   z,   w)
    const auto q2 = traits::add(q, q);         // (2x,  2y,  2z,  2w)
    const auto sq2 = traits::mul(q, q2);       // (2xx, 2yy, 2zz, 2ww)

    // Diagonal terms: (1 - 2yy - 2zz, 1 - 2xx - 2zz, 1 - 2xx - 2yy, 0)
    auto diagonal = traits::set4(T{ 1 }, T{ 1 }, T{ 1 }, T{ 0 });
    diagonal = traits::sub(diagonal, traits::template permute<1, 0, 0, 3>(sq2));   // (2yy, 2xx, 2xx, _)
    diagonal = traits::sub(diagonal, traits::template permute<2, 2, 1, 3>(sq2));   // (2zz, 2zz, 2yy, _)
    diagonal = traits::template blend<0b1000>(diagonal, zero);

    // Off-diagonal terms: 2xz, 2xy, 2yz and w * (2y, 2z, 2x), added or subtracted
    const auto products = traits::mul(traits::template permute<0, 0, 1, 3>(q), traits::template permute<2, 1, 2, 3>(q2));
    const auto w_terms = traits::mul(traits::template permute<3, 3, 3, 3>(q), traits::template permute<1, 2, 0, 3>(q2));

    const auto plus = traits::add(products, w_terms);    // (2xz + 2wy, 2xy + 2wz, 2yz + 2wx, _)
    const auto minus = traits::sub(products, w_terms);   // (2xz - 2wy, 2xy - 2wz, 2yz - 2wx, _)

    // Each row takes its lanes from diagonal / plus / minus
    // row0 = (diagonal[0], minus[1], plus[0], 0)
    const auto row0 = traits::template blend<0b0100>(
        traits::template blend<0b0010>(diagonal, minus),
        traits::template permute<0, 0, 0, 0>(plus));

    // row1 = (plus[1], diagonal[1], minus[2], 0)
    const auto row1 = traits::template blend<0b0100>(
        traits::template blend<0b0001>(diagonal, traits::template permute<1, 1, 1, 1>(plus)),
        minus);

    // row2 = (minus[0], plus[2], diagonal[2], 0)
    const auto row2 = traits::template blend<0b0010>(
        traits::template blend<0b0001>(diagonal, minus),
        traits::template permute<2, 2, 2, 2>(plus));

    // row3 = (0, 0, 0, 1)
    const auto row3 = traits::set4(T{ 0 }, T{ 0 }, T{ 0 }, T{ 1 });

    MatSIMD result;

    traits::store(result.Data() + 0, row0);
    traits::store(result.Data() + 4, row1);
    traits::store(result.Data() + 8, row2);
    traits::store(result.Data() + 12, row3);

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::TRS(const VecSIMD<U, 3>& translation,
                                                                                        const QuaternionSIMD<U>& rotation,
                                                                                        const VecSIMD<U, 3>& scale) {
    return Translate(translation) * Rotate(rotation) * Scale(scale);
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>
MatSIMD<T, RowCount, ColCount, Layout>::Perspective(U fovYRadians, U aspect, U nearPlane, U farPlane) {
    const U tanHalfFovY = std::tan(fovYRadians * U{ 0.5 });

    return MatSIMD{
        static_cast<T>(U{1} / (aspect * tanHalfFovY)), T{0}, T{0}, T{0},
        T{0}, static_cast<T>(U{1} / tanHalfFovY), T{0}, T{0},
        T{0}, T{0}, static_cast<T>(-(farPlane + nearPlane) / (farPlane - nearPlane)),
        static_cast<T>(-(U{2} *farPlane * nearPlane) / (farPlane - nearPlane)),
        T{0}, T{0}, static_cast<T>(-U{1}), T{0} };
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>
MatSIMD<T, RowCount, ColCount, Layout>::Ortho(U left, U right, U bottom, U top, U nearPlane, U farPlane) {
    return MatSIMD{
        static_cast<T>(U{2} / (right - left)), T{0}, T{0}, static_cast<T>(-(right + left) / (right - left)),
        T{0}, static_cast<T>(U{2} / (top - bottom)), T{0}, static_cast<T>(-(top + bottom) / (top - bottom)),
        T{0}, T{0}, static_cast<T>(-U{2} / (farPlane - nearPlane)), static_cast<T>(-(farPlane + nearPlane) / (farPlane - nearPlane)),
        T{0}, T{0}, T{0}, T{1} };
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>
MatSIMD<T, RowCount, ColCount, Layout>::LookAt(const VecSIMD<U, 3>& eye, const VecSIMD<U, 3>& target, const VecSIMD<U, 3>& up) {
    const VecSIMD<U, 3> forward = (target - eye).Normalized();
    const VecSIMD<U, 3> right = Cross(forward, up).Normalized();
    const VecSIMD<U, 3> trueUp = Cross(right, forward);

    return MatSIMD{
        static_cast<T>(right[0]), static_cast<T>(right[1]), static_cast<T>(right[2]), static_cast<T>(-Dot(right, eye)),
        static_cast<T>(trueUp[0]), static_cast<T>(trueUp[1]), static_cast<T>(trueUp[2]), static_cast<T>(-Dot(trueUp, eye)),
        static_cast<T>(-forward[0]), static_cast<T>(-forward[1]), static_cast<T>(-forward[2]), static_cast<T>(Dot(forward, eye)),
        T{0}, T{0}, T{0}, T{1} };
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr VecSIMD<std::common_type_t<T, U>, 3>
MatSIMD<T, RowCount, ColCount, Layout>::MultiplyPoint(const VecSIMD<U, 3>& point) const {
    using Result = std::common_type_t<T, U>;

    // A point is (x, y, z, 1): the translation column is added
    const VecSIMD<Result, 4> homogeneous(static_cast<Result>(point[0]), static_cast<Result>(point[1]), static_cast<Result>(point[2]), Result{ 1 });
	const VecSIMD<Result, 4> transformed = (*this) * homogeneous; // operator* here kills performance

    return VecSIMD<Result, 3>(transformed[0], transformed[1], transformed[2]);
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr VecSIMD<std::common_type_t<T, U>, 3>
MatSIMD<T, RowCount, ColCount, Layout>::MultiplyVector(const VecSIMD<U, 3>& vector) const {
    using Result = std::common_type_t<T, U>;

    // A direction is (x, y, z, 0): the translation column drops out
    const VecSIMD<Result, 4> homogeneous(static_cast<Result>(vector[0]), static_cast<Result>(vector[1]), static_cast<Result>(vector[2]), Result{ 0 });
    const VecSIMD<Result, 4> transformed = (*this) * homogeneous;

    return VecSIMD<Result, 3>(transformed[0], transformed[1], transformed[2]);
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr VecSIMD<T, 3> MatSIMD<T, RowCount, ColCount, Layout>::ExtractPosition() const {
    // Three elements: nothing to gain from SIMD here
    return VecSIMD<T, 3>((*this)[0, 3], (*this)[1, 3], (*this)[2, 3]);
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr VecSIMD<T, 3> MatSIMD<T, RowCount, ColCount, Layout>::ExtractScale() const {
    // The scale is the length of the first three columns
    const VecSIMD<T, 3> column0((*this)[0, 0], (*this)[1, 0], (*this)[2, 0]);
    const VecSIMD<T, 3> column1((*this)[0, 1], (*this)[1, 1], (*this)[2, 1]);
    const VecSIMD<T, 3> column2((*this)[0, 2], (*this)[1, 2], (*this)[2, 2]);

    return VecSIMD<T, 3>(column0.Length(), column1.Length(), column2.Length());
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr QuaternionSIMD<U> MatSIMD<T, RowCount, ColCount, Layout>::ExtractRotation() const {
    const VecSIMD<T, 3> scale = ExtractScale();

    const T sx = scale[0] != T{0} ? scale[0] : T{1};
    const T sy = scale[1] != T{0} ? scale[1] : T{1};
    const T sz = scale[2] != T{0} ? scale[2] : T{1};

    const T m00 = (*this)[0, 0] / sx;
    const T m10 = (*this)[1, 0] / sx;
    const T m20 = (*this)[2, 0] / sx;
    const T m01 = (*this)[0, 1] / sy;
    const T m11 = (*this)[1, 1] / sy;
    const T m21 = (*this)[2, 1] / sy;
    const T m02 = (*this)[0, 2] / sz;
    const T m12 = (*this)[1, 2] / sz;
    const T m22 = (*this)[2, 2] / sz;

    const T trace = m00 + m11 + m22;

    if (trace > T{0}) {
        const T s = std::sqrt(trace + T{1}) * T{2};
        return QuaternionSIMD<T>((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, s * T{0.25});
    }

    if (m00 > m11 && m00 > m22) {
        const T s = std::sqrt(T{1} + m00 - m11 - m22) * T{2};
        return QuaternionSIMD<T>(s * T{0.25}, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s);
    }

    if (m11 > m22) {
        const T s = std::sqrt(T{1} + m11 - m00 - m22) * T{2};
        return QuaternionSIMD<T>((m01 + m10) / s, s * T{0.25}, (m12 + m21) / s, (m02 - m20) / s);
    }

    const T s = std::sqrt(T{1} + m22 - m00 - m11) * T{2};
    return QuaternionSIMD<T>((m02 + m20) / s, (m12 + m21) / s, s * T{0.25}, (m10 - m01) / s);
}

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator+(const MatSIMD<T, RowCount, ColCount, Layout>& a, const MatSIMD<U, RowCount, ColCount, Layout>& b)
-> MatSIMDCommon<T, U, RowCount, ColCount, Layout> {
    using R = std::common_type_t<T, U>;

    constexpr std::size_t total_elements = RowCount * ColCount;

    MatSIMDCommon<T, U, RowCount, ColCount, Layout> result;

    if constexpr (std::same_as<T, U>) {
        Detail::ForEachBatch<R>(total_elements, [&]<typename traits>(std::size_t i) {
            auto a_vec = traits::load(a.Data() + i);
            auto b_vec = traits::load(b.Data() + i);
            auto res_vec = traits::add(a_vec, b_vec);

            traits::store(result.Data() + i, res_vec);
        });
    }
    else {
        std::println("SIMD add fallback");
        for (std::size_t row = 0; row < RowCount; ++row) {
            for (std::size_t col = 0; col < ColCount; ++col) {
                result[row, col] = static_cast<R>(a[row, col]) + static_cast<R>(b[row, col]);
            }
        }
    }

    return result;
}

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator-(const MatSIMD<T, RowCount, ColCount, Layout>& a, const MatSIMD<U, RowCount, ColCount, Layout>& b)
-> MatSIMDCommon<T, U, RowCount, ColCount, Layout> {
    using R = std::common_type_t<T, U>;

    constexpr std::size_t total_elements = RowCount * ColCount;

    MatSIMDCommon<T, U, RowCount, ColCount, Layout> result;

    if constexpr (std::same_as<T, U>) {
        Detail::ForEachBatch<R>(total_elements, [&]<typename traits>(std::size_t i) {
            auto a_vec = traits::load(a.Data() + i);
            auto b_vec = traits::load(b.Data() + i);
            auto res_vec = traits::sub(a_vec, b_vec);

            traits::store(result.Data() + i, res_vec);
        });
    }
    else {
        std::println("SIMD sub fallback");
        for (std::size_t row = 0; row < RowCount; ++row) {
            for (std::size_t col = 0; col < ColCount; ++col) {
                result[row, col] = static_cast<R>(a[row, col]) - static_cast<R>(b[row, col]);
            }
        }
    }

    return result;
}

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator*(const MatSIMD<T, RowCount, ColCount, Layout>& mat, U scalar)
-> MatSIMDCommon<T, U, RowCount, ColCount, Layout> {
    using R = std::common_type_t<T, U>;

    constexpr std::size_t total_elements = RowCount * ColCount;

    MatSIMDCommon<T, U, RowCount, ColCount, Layout> result;

    if constexpr (std::same_as<T, R>) {
        Detail::ForEachBatch<R>(total_elements, [&]<typename traits>(std::size_t i) {
            auto data_vec = traits::load(mat.Data() + i);
            auto scalar_vec = traits::set1(static_cast<R>(scalar));
            auto res_vec = traits::mul(data_vec, scalar_vec);

            traits::store(result.Data() + i, res_vec);
        });
    }
    else {
        std::println("SIMD mul fallback");
        for (std::size_t row = 0; row < RowCount; ++row) {
            for (std::size_t col = 0; col < ColCount; ++col) {
                result[row, col] = static_cast<R>(mat[row, col]) * static_cast<R>(scalar);
            }
        }
    }

    return result;
}

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator*(U scalar, const MatSIMD<T, RowCount, ColCount, Layout>& mat)
-> MatSIMDCommon<T, U, RowCount, ColCount, Layout> {
    return mat * scalar;
}

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator/(const MatSIMD<T, RowCount, ColCount, Layout>& mat, U scalar)
-> MatSIMDCommon<T, U, RowCount, ColCount, Layout> {
    using R = std::common_type_t<T, U>;

    constexpr std::size_t total_elements = RowCount * ColCount;

    MatSIMDCommon<T, U, RowCount, ColCount, Layout> result;

    if constexpr (std::same_as<T, R>) {
        Detail::ForEachBatch<R>(total_elements, [&]<typename traits>(std::size_t i) {
            auto data_vec = traits::load(mat.Data() + i);
            auto scalar_vec = traits::set1(static_cast<R>(scalar));
            auto res_vec = traits::div(data_vec, scalar_vec);

            traits::store(result.Data() + i, res_vec);
        });
    }
    else {
        std::println("SIMD div fallback");
        for (std::size_t row = 0; row < RowCount; ++row) {
            for (std::size_t col = 0; col < ColCount; ++col) {
                result[row, col] = static_cast<R>(mat[row, col]) / static_cast<R>(scalar);
            }
        }
    }

    return result;
}

template <float_num T, float_num U, std::size_t A, std::size_t B, std::size_t C, typename Layout>
inline constexpr auto operator*(const MatSIMD<T, A, B, Layout>& lhs, const MatSIMD<U, B, C, Layout>& rhs)
-> MatSIMDCommon<T, U, A, C, Layout> {
    using R = std::common_type_t<T, U>;

    MatSIMDCommon<T, U, A, C, Layout> result;

    if constexpr (std::same_as<T, U> && std::same_as<Layout, ColumnMajor>) {
        // Column `col` of the result = sum over i of rhs(i, col) * (column i of lhs).
        // Columns are contiguous, so the batches run along the A rows.
        for (std::size_t col = 0; col < C; ++col) {
            Detail::ForEachBatch<R>(A, [&]<typename traits>(std::size_t j) {
                auto acc = traits::set1(R{});

                for (std::size_t i = 0; i < B; ++i) {
                    auto rhs_vec = traits::set1(rhs.Data()[i + col * B]);
                    auto column_vec = traits::load(lhs.Data() + i * A + j);

                    acc = traits::add(acc, traits::mul(rhs_vec, column_vec));
                }

                traits::store(result.Data() + col * A + j, acc);
            });
        }
    }
    else if constexpr (std::same_as<T, U>) {
        // Row `row` of the result = sum over i of lhs(row, i) * (row i of rhs).
        // Rows are contiguous, so the batches run along the C columns.
        for (std::size_t row = 0; row < A; ++row) {
            Detail::ForEachBatch<R>(C, [&]<typename traits>(std::size_t j) {
                auto acc = traits::set1(R{});

                for (std::size_t i = 0; i < B; ++i) {
                    auto lhs_vec = traits::set1(lhs.Data()[row * B + i]);
                    auto row_vec = traits::load(rhs.Data() + i * C + j);

                    acc = traits::add(acc, traits::mul(lhs_vec, row_vec));
                }

                traits::store(result.Data() + row * C + j, acc);
            });
        }
    }
    else {
        std::println("SIMD mul fallback");
        for (std::size_t row = 0; row < A; ++row) {
            for (std::size_t col = 0; col < C; ++col) {
                R sum{};

                for (std::size_t i = 0; i < B; ++i) {
                    sum += static_cast<R>(lhs[row, i]) * static_cast<R>(rhs[i, col]);
                }

                result[row, col] = sum;
            }
        }
    }

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename FromLayout, typename ToLayout>
inline constexpr MatSIMD<T, RowCount, ColCount, ToLayout> MatCastLayout(const MatSIMD<T, RowCount, ColCount, FromLayout>& mat) {
    constexpr std::size_t total_elements = RowCount * ColCount;

    MatSIMD<T, RowCount, ColCount, ToLayout> result;

    if constexpr (std::same_as<FromLayout, ToLayout>) {
        Detail::ForEachBatch<T>(total_elements, [&]<typename traits>(std::size_t i) {
            traits::store(result.Data() + i, traits::load(mat.Data() + i));
        });
    }
    else {
        // Slot k of the result takes mat.Data()[source_index[k]]: same (row, col), other position in memory
        constexpr auto source_index = []() {
            typename FromLayout::template mapping<std::extents<std::size_t, RowCount, ColCount>> from_mapping{};
            typename ToLayout::template mapping<std::extents<std::size_t, RowCount, ColCount>> to_mapping{};
            std::array<std::int32_t, total_elements> table{};

            for (std::size_t row = 0; row < RowCount; ++row) {
                for (std::size_t col = 0; col < ColCount; ++col) {
                    table[to_mapping(row, col)] = static_cast<std::int32_t>(from_mapping(row, col));
                }
            }

            return table;
            }();

        Detail::ForEachBatch<T>(total_elements, [&]<typename traits>(std::size_t i) {
            traits::store(result.Data() + i, traits::gather(mat.Data(), source_index.data() + i));
        });
    }

    return result;
}

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator*(const MatSIMD<T, RowCount, ColCount, Layout>& mat, const VecSIMD<U, ColCount>& vec)
-> VecSIMD<std::common_type_t<T, U>, RowCount> {
    using R = std::common_type_t<T, U>;

    std::array<R, RowCount> out{};

    if constexpr (std::same_as<T, U> && std::same_as<Layout, ColumnMajor>) {
        // result = sum over col of vec[col] * (column col of mat). Columns are contiguous,
        // so the batches run along the rows.
        Detail::ForEachBatch<R>(RowCount, [&]<typename traits>(std::size_t j) {
            auto acc = traits::set1(R{});

            for (std::size_t col = 0; col < ColCount; ++col) {
                auto vec_vec = traits::set1(vec[col]);
                auto column_vec = traits::load(mat.Data() + col * RowCount + j);

                acc = traits::add(acc, traits::mul(vec_vec, column_vec));
            }

            traits::store(out.data() + j, acc);
        });
    }
    else if constexpr (std::same_as<T, U>) {
        // result[row] = dot(row of mat, vec). Rows are contiguous, so the batches run along the columns
        // and each batch is summed horizontally.
        std::array<R, ColCount> in{};

        for (std::size_t col = 0; col < ColCount; ++col) {
            in[col] = vec[col];
        }

        for (std::size_t row = 0; row < RowCount; ++row) {
            R sum{};

            Detail::ForEachBatch<R>(ColCount, [&]<typename traits>(std::size_t j) {
                auto row_vec = traits::load(mat.Data() + row * ColCount + j);
                auto vec_vec = traits::load(in.data() + j);

                sum += traits::reduce_add(traits::mul(row_vec, vec_vec));
            });

            out[row] = sum;
        }
    }
    else {
        std::println("SIMD mul fallback");
        for (std::size_t row = 0; row < RowCount; ++row) {
            for (std::size_t col = 0; col < ColCount; ++col) {
                out[row] += static_cast<R>(mat[row, col]) * static_cast<R>(vec[col]);
            }
        }
    }

    VecSIMD<R, RowCount> result;

    for (std::size_t row = 0; row < RowCount; ++row) {
        result[row] = out[row];
    }

    return result;
}

template <float_num T>
inline constexpr T ToRadiansSIMD(T degrees) {
    return degrees * (std::numbers::pi_v<T> / T{ 180 });
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr bool ValidTRS(const MatSIMD<T, RowCount, ColCount, Layout>& mat) {
    if constexpr (RowCount != 4 || ColCount != 4) {
        return false;
    } else {
        constexpr T kEpsilon = static_cast<T>(1e-5);

        if (mat[3, 0] != T{} || mat[3, 1] != T{} || mat[3, 2] != T{}) {
            return false;
        }

        if (mat[3, 3] < T{1} - kEpsilon || mat[3, 3] > T{1} + kEpsilon) {
            return false;
        }

        // Every axis must have a length
        const VecSIMD<T, 3> scale = mat.ExtractScale();

        if (scale[0] <= T{} || scale[1] <= T{} || scale[2] <= T{}) {
            return false;
        }

        return true;
    }
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
struct std::formatter<MatSIMD<T, RowCount, ColCount, Layout>> {
    std::formatter<T> underlying;

    constexpr auto parse(std::format_parse_context& ctx) {
        return underlying.parse(ctx);
    }

    auto format(const MatSIMD<T, RowCount, ColCount, Layout>& obj, std::format_context& ctx) const {
        auto out = ctx.out();

        out = std::format_to(out, "[");

        for (std::size_t row = 0; row < RowCount; ++row) {
            if (row != 0) {
                out = std::format_to(out, ", ");
            }

            out = std::format_to(out, "[");

            for (std::size_t col = 0; col < ColCount; ++col) {
                if (col != 0) {
                    out = std::format_to(out, ", ");
                }

                out = underlying.format(obj[row, col], ctx);
            }

            out = std::format_to(out, "]");
        }

        return std::format_to(out, "]");
    }
};
