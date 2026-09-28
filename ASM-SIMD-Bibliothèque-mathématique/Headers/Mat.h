#include <array>
#include <concepts>
#include <cstdint>
#include <format>
#include <mdspan>
#include <numbers>

#include "Vec.h"
#include "Quaternion.h"

using RowMajor = std::layout_right;
using ColumnMajor = std::layout_left;

template <std::floating_point T, std::size_t RowCount, std::size_t ColCount, typename Layout = RowMajor>
class Mat {
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
    constexpr Mat() = default;

    template <typename... Args>
        requires(sizeof...(Args) == RowCount * ColCount) && (std::convertible_to<Args, T> && ...)
    constexpr Mat(Args&&... args);

    template <std::floating_point U, typename OtherLayout>
    constexpr Mat(const Mat<U, RowCount, ColCount, OtherLayout>& other);

    constexpr T& operator[](size_type row, size_type col);
    constexpr const T& operator[](size_type row, size_type col) const;

    constexpr T& operator()(size_type row, size_type col);
    constexpr const T& operator()(size_type row, size_type col) const;

    constexpr T* Data();
    constexpr const T* Data() const;

    constexpr auto MDSpan();
    constexpr auto MDSpan() const;

    static constexpr size_type Rows();
    static constexpr size_type Cols();
    static constexpr size_type Size();

    template <std::floating_point U>
    constexpr bool operator==(const Mat<U, RowCount, ColCount, Layout>& other) const;

    template <std::floating_point U, typename OtherLayout>
    constexpr bool operator==(const Mat<U, RowCount, ColCount, OtherLayout>& other) const;

    template <std::floating_point U, typename OtherLayout>
    constexpr Mat& operator+=(const Mat<U, RowCount, ColCount, OtherLayout>& other);

    template <std::floating_point U, typename OtherLayout>
    constexpr Mat& operator-=(const Mat<U, RowCount, ColCount, OtherLayout>& other);

    template <std::floating_point U>
    constexpr Mat& operator*=(U scalar);

    template <std::floating_point U>
    constexpr Mat& operator/=(U scalar);

    constexpr Mat operator+() const;
    constexpr Mat operator-() const;

    constexpr Mat<T, ColCount, RowCount, Layout> Transpose() const;

    template <std::size_t N = RowCount>
        requires(RowCount == ColCount && N == RowCount)
    static constexpr Mat Identity();

    template <std::size_t N = RowCount>
        requires(RowCount == ColCount && N == RowCount)
    constexpr auto Determinant() const;

    template <std::size_t N = RowCount>
        requires(RowCount == ColCount && N == RowCount)
    constexpr Mat Inverse() const;

    template <std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount)
    static constexpr Mat Zero();

    template <std::floating_point U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static constexpr Mat RotationX(U angleRadians);

    template <std::floating_point U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static constexpr Mat RotationY(U angleRadians);

    template <std::floating_point U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static constexpr Mat RotationZ(U angleRadians);

    template <std::floating_point U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static constexpr Mat Scale(const Vec3<U>& scale);

    template <std::floating_point U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static constexpr Mat Translate(const Vec3<U>& translation);

    template <std::floating_point U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static constexpr Mat Rotate(const Quaternion<U>& rotation);

    template <std::floating_point U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static constexpr Mat TRS(const Vec3<U>& translation, const Quaternion<U>& rotation, const Vec3<U>& scale);

    template <std::floating_point U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static constexpr Mat Perspective(U fovYRadians, U aspect, U nearPlane, U farPlane);

    template <std::floating_point U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static constexpr Mat Ortho(U left, U right, U bottom, U top, U nearPlane, U farPlane);

    template <std::floating_point U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    static constexpr Mat LookAt(const Vec3<U>& eye, const Vec3<U>& target, const Vec3<U>& up);

    template <std::floating_point U, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    constexpr Vec3<std::common_type_t<T, U>> MultiplyPoint(const Vec3<U>& point) const;

    template <std::floating_point U, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    constexpr Vec3<std::common_type_t<T, U>> MultiplyVector(const Vec3<U>& vector) const;

    template <std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    constexpr Vec3<T> ExtractPosition() const;

    template <std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    constexpr Vec3<T> ExtractScale() const;

    template <std::floating_point U = T, std::size_t R = RowCount, std::size_t C = ColCount>
        requires(R == RowCount && C == ColCount && R == 4 && C == 4)
    constexpr Quaternion<U> ExtractRotation() const;
};

template <std::floating_point T, std::floating_point U, std::size_t RowCount, std::size_t ColCount, typename Layout>
using MatCommon = Mat<std::common_type_t<T, U>, RowCount, ColCount, Layout>;

template <std::floating_point T, std::floating_point U, std::size_t RowCount, std::size_t ColCount, typename Layout>
constexpr auto operator+(const Mat<T, RowCount, ColCount, Layout>& a, const Mat<U, RowCount, ColCount, Layout>& b)
    -> MatCommon<T, U, RowCount, ColCount, Layout>;

template <std::floating_point T, std::floating_point U, std::size_t RowCount, std::size_t ColCount, typename Layout>
constexpr auto operator-(const Mat<T, RowCount, ColCount, Layout>& a, const Mat<U, RowCount, ColCount, Layout>& b)
    -> MatCommon<T, U, RowCount, ColCount, Layout>;

template <std::floating_point T, std::floating_point U, std::size_t RowCount, std::size_t ColCount, typename Layout>
constexpr auto operator*(const Mat<T, RowCount, ColCount, Layout>& mat, U scalar)
    -> MatCommon<T, U, RowCount, ColCount, Layout>;

template <std::floating_point T, std::floating_point U, std::size_t RowCount, std::size_t ColCount, typename Layout>
constexpr auto operator*(U scalar, const Mat<T, RowCount, ColCount, Layout>& mat)
    -> MatCommon<T, U, RowCount, ColCount, Layout>;

template <std::floating_point T, std::floating_point U, std::size_t RowCount, std::size_t ColCount, typename Layout>
constexpr auto operator/(const Mat<T, RowCount, ColCount, Layout>& mat, U scalar)
    -> MatCommon<T, U, RowCount, ColCount, Layout>;

// (A x B) * (B x C) = (A x C)
template <std::floating_point T, std::floating_point U, std::size_t A, std::size_t B, std::size_t C, typename Layout>
constexpr auto operator*(const Mat<T, A, B, Layout>& lhs, const Mat<U, B, C, Layout>& rhs)
    -> MatCommon<T, U, A, C, Layout>;

template <std::floating_point T, std::size_t RowCount, std::size_t ColCount, typename FromLayout, typename ToLayout>
constexpr Mat<T, RowCount, ColCount, ToLayout> MatCastLayout(const Mat<T, RowCount, ColCount, FromLayout>& mat);

template <std::floating_point T, std::floating_point U, std::size_t RowCount, std::size_t ColCount, typename Layout>
constexpr auto operator*(const Mat<T, RowCount, ColCount, Layout>& mat, const Vec<U, ColCount>& vec)
    -> Vec<std::common_type_t<T, U>, RowCount>;

template <std::floating_point T>
constexpr T ToRadians(T degrees);

template <std::floating_point T, std::size_t RowCount, std::size_t ColCount, typename Layout>
constexpr bool ValidTRS(const Mat<T, RowCount, ColCount, Layout>& mat);

template <std::floating_point T, std::size_t RowCount, std::size_t ColCount>
using MatRowMajor = Mat<T, RowCount, ColCount, RowMajor>;

template <std::floating_point T, std::size_t RowCount, std::size_t ColCount>
using MatColumnMajor = Mat<T, RowCount, ColCount, ColumnMajor>;

template <std::floating_point T, std::size_t N>
using SquareMatrix = Mat<T, N, N>;

template <std::floating_point T, std::size_t N>
using SquareMatrixRowMajor = Mat<T, N, N, RowMajor>;

template <std::floating_point T, std::size_t N>
using SquareMatrixColumnMajor = Mat<T, N, N, ColumnMajor>;

template <std::floating_point T>
using Mat2 = Mat<T, 2, 2>;

template <std::floating_point T>
using Mat2ColumnMajor = Mat<T, 2, 2, ColumnMajor>;

template <std::floating_point T>
using Mat3 = Mat<T, 3, 3>;

template <std::floating_point T>
using Mat3ColumnMajor = Mat<T, 3, 3, ColumnMajor>;

template <std::floating_point T>
using Mat4 = Mat<T, 4, 4>;

template <std::floating_point T>
using Mat4ColumnMajor = Mat<T, 4, 4, ColumnMajor>;

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

template <std::floating_point T, std::size_t RowCount, std::size_t ColCount, typename Layout>
struct std::formatter<Mat<T, RowCount, ColCount, Layout>>;

#include "Mat.inl"
