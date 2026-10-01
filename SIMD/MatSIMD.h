#include <array>
#include <concepts>
#include <cstdint>
#include <format>
#include <mdspan>
#include <numbers>

#include "VecSIMD.h"

#include "Concepts.h"

using RowMajor = std::layout_right;
using ColumnMajor = std::layout_left;

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout = RowMajor>
class MatSIMD {
public:
    using value_type = T;
    using size_type = std::size_t;
    using layout_type = Layout;

    using extents_type = std::extents<size_type, RowCount, ColCount>;
    using mapping_type = Layout::template mapping<extents_type>;
    using mdspan_type = std::mdspan<T, extents_type, Layout>;
    using const_mdspan_type = std::mdspan<const T, extents_type, Layout>;

private:
    static constexpr extents_type m_extents{};
    static constexpr mapping_type m_mapping{m_extents};
    std::array<T, RowCount * ColCount> m_data{};

public:
    inline constexpr MatSIMD() = default;

    template <typename... Args>
        requires(sizeof...(Args) == RowCount * ColCount) && (std::convertible_to<Args, T> && ...)
    inline constexpr MatSIMD(Args&&... args);

    template <float_num U, typename OtherLayout>
    inline constexpr MatSIMD(const MatSIMD<U, RowCount, ColCount, OtherLayout>& other);

    inline constexpr T& operator[](size_type row, size_type col);
    inline constexpr const T& operator[](size_type row, size_type col) const;

    inline constexpr T& operator()(size_type row, size_type col);
    inline constexpr const T& operator()(size_type row, size_type col) const;

    inline constexpr T* Data();
    inline constexpr const T* Data() const;

    inline constexpr auto MDSpan();
    inline constexpr auto MDSpan() const;

    static inline constexpr size_type Rows();
    static inline constexpr size_type Cols();
    static inline constexpr size_type Size();

    template <float_num U>
    inline constexpr bool operator==(const MatSIMD<U, RowCount, ColCount, Layout>& other) const;

    template <float_num U, typename OtherLayout>
    inline constexpr bool operator==(const MatSIMD<U, RowCount, ColCount, OtherLayout>& other) const;

    template <float_num U, typename OtherLayout>
    inline constexpr MatSIMD& operator+=(const MatSIMD<U, RowCount, ColCount, OtherLayout>& other);

    template <float_num U, typename OtherLayout>
    inline constexpr MatSIMD& operator-=(const MatSIMD<U, RowCount, ColCount, OtherLayout>& other);

    template <float_num U>
    inline constexpr MatSIMD& operator*=(U scalar);

    template <float_num U>
    inline constexpr MatSIMD& operator/=(U scalar);

    inline constexpr MatSIMD operator+() const;
    inline constexpr MatSIMD operator-() const;

    inline constexpr MatSIMD<T, ColCount, RowCount, Layout> Transpose() const;

    template <std::size_t N = RowCount>
        requires(RowCount == ColCount && N == RowCount)
    static inline constexpr MatSIMD Identity();

    template <std::size_t N = RowCount>
        requires(RowCount == ColCount && N == RowCount)
    inline constexpr auto Determinant() const;

    template <std::size_t N = RowCount>
        requires(RowCount == ColCount && N == RowCount)
    inline constexpr MatSIMD Inverse() const;

    template <std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount)
    static inline constexpr MatSIMD Zero();

    template <float_num U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static inline constexpr MatSIMD RotationX(U angleRadians);

    template <float_num U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static inline constexpr MatSIMD RotationY(U angleRadians);

    template <float_num U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static inline constexpr MatSIMD RotationZ(U angleRadians);

    template <float_num U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static inline constexpr MatSIMD Scale(const VecSIMD<U, 3>& scale);

    template <float_num U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static inline constexpr MatSIMD Translate(const VecSIMD<U, 3>& translation);

    //template <float_num U = T, std::size_t R = RowCount, std::size_t C = ColCount>
    //    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    //static inline constexpr MatSIMD Rotate(const QuaternionSIMD<U>& rotation);

    //template <float_num U = T, std::size_t R = RowCount, std::size_t C = ColCount>
    //    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    //static inline constexpr MatSIMD TRS(const VecSIMD<U, 3>& translation, const QuaternionSIMD<U>& rotation, const VecSIMD<U, 3>& scale);

    template <float_num U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static inline constexpr MatSIMD Perspective(U fovYRadians, U aspect, U nearPlane, U farPlane);

    template <float_num U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static inline constexpr MatSIMD Ortho(U left, U right, U bottom, U top, U nearPlane, U farPlane);

    template <float_num U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static inline constexpr MatSIMD LookAt(const VecSIMD<U, 3>& eye, const VecSIMD<U, 3>& target, const VecSIMD<U, 3>& up);

    template <float_num U, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    inline constexpr VecSIMD<std::common_type_t<T, U>, 3> MultiplyPoint(const VecSIMD<U, 3>& point) const;

    template <float_num U, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    inline constexpr VecSIMD<std::common_type_t<T, U> , 3> MultiplyVector(const VecSIMD<U, 3>& vector) const;

    template <std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    inline constexpr VecSIMD<T, 3> ExtractPosition() const;

    template <std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    inline constexpr VecSIMD<T, 3> ExtractScale() const;

    //template <float_num U = T, std::size_t R = RowCount, std::size_t C = ColCount>
    //    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    //inline constexpr QuaternionSIMD<U> ExtractRotation() const;
};

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
using MatSIMDCommon = MatSIMD<std::common_type_t<T, U>, RowCount, ColCount, Layout>;

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator+(const MatSIMD<T, RowCount, ColCount, Layout>& a, const MatSIMD<U, RowCount, ColCount, Layout>& b)
    -> MatSIMDCommon<T, U, RowCount, ColCount, Layout>;

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator-(const MatSIMD<T, RowCount, ColCount, Layout>& a, const MatSIMD<U, RowCount, ColCount, Layout>& b)
    -> MatSIMDCommon<T, U, RowCount, ColCount, Layout>;

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator*(const MatSIMD<T, RowCount, ColCount, Layout>& mat, U scalar)
    -> MatSIMDCommon<T, U, RowCount, ColCount, Layout>;

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator*(U scalar, const MatSIMD<T, RowCount, ColCount, Layout>& mat)
    -> MatSIMDCommon<T, U, RowCount, ColCount, Layout>;

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator/(const MatSIMD<T, RowCount, ColCount, Layout>& mat, U scalar)
    -> MatSIMDCommon<T, U, RowCount, ColCount, Layout>;

// (A x B) * (B x C) = (A x C)
template <float_num T, float_num U, std::size_t A, std::size_t B, std::size_t C, typename Layout>
inline constexpr auto operator*(const MatSIMD<T, A, B, Layout>& lhs, const MatSIMD<U, B, C, Layout>& rhs)
    -> MatSIMDCommon<T, U, A, C, Layout>;

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename FromLayout, typename ToLayout>
inline constexpr MatSIMD<T, RowCount, ColCount, ToLayout> MatCastLayout(const MatSIMD<T, RowCount, ColCount, FromLayout>& mat);

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator*(const MatSIMD<T, RowCount, ColCount, Layout>& mat, const VecSIMD<U, ColCount>& vec)
    -> VecSIMD<std::common_type_t<T, U>, RowCount>;

template <float_num T>
inline constexpr T ToRadians(T degrees);

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr bool ValidTRS(const MatSIMD<T, RowCount, ColCount, Layout>& mat);

template <float_num T, std::size_t RowCount, std::size_t ColCount>
using MatRowMajor = MatSIMD<T, RowCount, ColCount, RowMajor>;

template <float_num T, std::size_t RowCount, std::size_t ColCount>
using MatColumnMajor = MatSIMD<T, RowCount, ColCount, ColumnMajor>;

template <float_num T, std::size_t N>
using SquareMatrix = MatSIMD<T, N, N>;

template <float_num T, std::size_t N>
using SquareMatrixRowMajor = MatSIMD<T, N, N, RowMajor>;

template <float_num T, std::size_t N>
using SquareMatrixColumnMajor = MatSIMD<T, N, N, ColumnMajor>;

template <float_num T>
using Mat2 = MatSIMD<T, 2, 2>;

template <float_num T>
using Mat2ColumnMajor = MatSIMD<T, 2, 2, ColumnMajor>;

template <float_num T>
using Mat3 = MatSIMD<T, 3, 3>;

template <float_num T>
using Mat3ColumnMajor = MatSIMD<T, 3, 3, ColumnMajor>;

template <float_num T>
using Mat4 = MatSIMD<T, 4, 4>;

template <float_num T>
using Mat4ColumnMajor = MatSIMD<T, 4, 4, ColumnMajor>;

using Mat2f = Mat2<float>;
using Mat3f = Mat3<float>;
using Mat4f = Mat4<float>;

using Mat2fColumnMajor = Mat2ColumnMajor<float>;
using Mat3fColumnMajor = Mat3ColumnMajor<float>;
using Mat4fColumnMajor = Mat4ColumnMajor<float>;

using Mat2d = Mat2<double>;
using Mat3d = Mat3<double>;
using Mat4d = Mat4<double>;

using Mat2dColumnMajor = Mat2ColumnMajor<double>;
using Mat3dColumnMajor = Mat3ColumnMajor<double>;
using Mat4dColumnMajor = Mat4ColumnMajor<double>;

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
struct std::formatter<MatSIMD<T, RowCount, ColCount, Layout>>;

#include "MatSIMD.inl"
