#pragma once

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <typename... Args>
    requires(sizeof...(Args) == RowCount * ColCount) && (std::convertible_to<Args, T> && ...)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>::MatSIMD(Args&&... args) {
    const std::array<T, RowCount * ColCount> values{static_cast<T>(std::forward<Args>(args))...};

    for (size_type row = 0; row < RowCount; ++row) {
        for (size_type col = 0; col < ColCount; ++col) {
            (*this)[row, col] = values[row * ColCount + col];
        }
    }
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, typename OtherLayout>
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>::MatSIMD(const MatSIMD<U, RowCount, ColCount, OtherLayout>& other) {
    for (size_type row = 0; row < RowCount; ++row) {
        for (size_type col = 0; col < ColCount; ++col) {
            (*this)[row, col] = static_cast<T>(other[row, col]);
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
    return mdspan_type{m_data.data()};
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto MatSIMD<T, RowCount, ColCount, Layout>::MDSpan() const {
    return const_mdspan_type{m_data.data()};
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
    for (size_type row = 0; row < RowCount; ++row) {
        for (size_type col = 0; col < ColCount; ++col) {
            if ((*this)[row, col] != other[row, col]) {
                return false;
            }
        }
    }

    return true;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, typename OtherLayout>
inline constexpr bool
MatSIMD<T, RowCount, ColCount, Layout>::operator==(const MatSIMD<U, RowCount, ColCount, OtherLayout>& other) const {
    for (size_type row = 0; row < RowCount; ++row) {
        for (size_type col = 0; col < ColCount; ++col) {
            if ((*this)[row, col] != other[row, col]) {
                return false;
            }
        }
    }

    return true;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, typename OtherLayout>
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>&
MatSIMD<T, RowCount, ColCount, Layout>::operator+=(const MatSIMD<U, RowCount, ColCount, OtherLayout>& other) {
    for (size_type row = 0; row < RowCount; ++row) {
        for (size_type col = 0; col < ColCount; ++col) {
            (*this)[row, col] += other[row, col];
        }
    }

    return *this;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, typename OtherLayout>
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>&
MatSIMD<T, RowCount, ColCount, Layout>::operator-=(const MatSIMD<U, RowCount, ColCount, OtherLayout>& other) {
    for (size_type row = 0; row < RowCount; ++row) {
        for (size_type col = 0; col < ColCount; ++col) {
            (*this)[row, col] -= other[row, col];
        }
    }

    return *this;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U>
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>& MatSIMD<T, RowCount, ColCount, Layout>::operator*=(U scalar) {
    for (auto& value : m_data) {
        value *= scalar;
    }

    return *this;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U>
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>& MatSIMD<T, RowCount, ColCount, Layout>::operator/=(U scalar) {
    for (auto& value : m_data) {
        value /= scalar;
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

    for (size_type row = 0; row < RowCount; ++row) {
        for (size_type col = 0; col < ColCount; ++col) {
            result[row, col] = -(*this)[row, col];
        }
    }

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr MatSIMD<T, ColCount, RowCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::Transpose() const {
    MatSIMD<T, ColCount, RowCount, Layout> result;

    for (size_type row = 0; row < RowCount; ++row) {
        for (size_type col = 0; col < ColCount; ++col) {
            result[col, row] = (*this)[row, col];
        }
    }

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <std::size_t N>
    requires(RowCount == ColCount && N == RowCount)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::Identity() {
    MatSIMD result;

    for (size_type i = 0; i < RowCount; ++i) {
        result[i, i] = T{1};
    }

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <std::size_t N>
    requires(RowCount == ColCount && N == RowCount)
inline constexpr auto MatSIMD<T, RowCount, ColCount, Layout>::Determinant() const {
    static_assert(RowCount == ColCount);

    MatSIMD<T, RowCount, ColCount, Layout> temp = *this;
    T result = T{1};

    for (size_type i = 0; i < RowCount; ++i) {
        const T pivot = temp[i, i];

        if (pivot == T{}) {
            return T{};
        }

        result *= pivot;

        for (size_type row = i + 1; row < RowCount; ++row) {
            const T factor = temp[row, i] / pivot;

            for (size_type col = i; col < ColCount; ++col) {
                temp[row, col] -= factor * temp[i, col];
            }
        }
    }

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <std::size_t N>
    requires(RowCount == ColCount && N == RowCount)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::Inverse() const {
    static_assert(RowCount == ColCount);

    MatSIMD<T, RowCount, ColCount * 2, Layout> augmented;

    for (size_type row = 0; row < RowCount; ++row) {
        for (size_type col = 0; col < ColCount; ++col) {
            augmented[row, col] = (*this)[row, col];
        }

        for (size_type col = 0; col < ColCount; ++col) {
            augmented[row, ColCount + col] = row == col ? T{1} : T{};
        }
    }

    for (size_type i = 0; i < RowCount; ++i) {
        const T pivot = augmented[i, i];

        if (pivot == T{}) {
            return MatSIMD{};
        }

        for (size_type col = 0; col < ColCount * 2; ++col) {
            augmented[i, col] /= pivot;
        }

        for (size_type row = 0; row < RowCount; ++row) {
            if (row == i) {
                continue;
            }

            const T factor = augmented[row, i];

            for (size_type col = 0; col < ColCount * 2; ++col) {
                augmented[row, col] -= factor * augmented[i, col];
            }
        }
    }

    MatSIMD result;

    for (size_type row = 0; row < RowCount; ++row) {
        for (size_type col = 0; col < ColCount; ++col) {
            result[row, col] = augmented[row, ColCount + col];
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

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::RotationX(U angleRadians) {
    const U s = std::sin(angleRadians);
    const U c = std::cos(angleRadians);

    MatSIMD result = Identity();
    result[1, 1] = static_cast<T>(c);
    result[1, 2] = static_cast<T>(-s);
    result[2, 1] = static_cast<T>(s);
    result[2, 2] = static_cast<T>(c);

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::RotationY(U angleRadians) {
    const U s = std::sin(angleRadians);
    const U c = std::cos(angleRadians);

    MatSIMD result = Identity();
    result[0, 0] = static_cast<T>(c);
    result[0, 2] = static_cast<T>(s);
    result[2, 0] = static_cast<T>(-s);
    result[2, 2] = static_cast<T>(c);

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::RotationZ(U angleRadians) {
    const U s = std::sin(angleRadians);
    const U c = std::cos(angleRadians);

    MatSIMD result = Identity();
    result[0, 0] = static_cast<T>(c);
    result[0, 1] = static_cast<T>(-s);
    result[1, 0] = static_cast<T>(s);
    result[1, 1] = static_cast<T>(c);

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::Scale(const VecSIMD<U, 3>& scale) {
    MatSIMD result = Identity();
    result[0, 0] = static_cast<T>(scale[0]);
    result[1, 1] = static_cast<T>(scale[1]);
    result[2, 2] = static_cast<T>(scale[2]);

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>
MatSIMD<T, RowCount, ColCount, Layout>::Translate(const VecSIMD<U, 3>& translation) {
    MatSIMD result = Identity();
    result[0, 3] = static_cast<T>(translation[0]);
    result[1, 3] = static_cast<T>(translation[1]);
    result[2, 3] = static_cast<T>(translation[2]);

    return result;
}

//template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
//template <float_num U, std::size_t R, std::size_t C>
//    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
//inline constexpr MatSIMD<T, RowCount, ColCount, Layout>
//MatSIMD<T, RowCount, ColCount, Layout>::Rotate(const QuaternionSIMD<U>& rotation) {
//    const U x = rotation.X();
//    const U y = rotation.Y();
//    const U z = rotation.Z();
//    const U w = rotation.W();
//
//    const U xx = x * x;
//    const U yy = y * y;
//    const U zz = z * z;
//    const U xy = x * y;
//    const U xz = x * z;
//    const U yz = y * z;
//    const U wx = w * x;
//    const U wy = w * y;
//    const U wz = w * z;
//
//    MatSIMD result = Identity();
//
//    result[0, 0] = static_cast<T>(U{1} - U{2} * (yy + zz));
//    result[0, 1] = static_cast<T>(U{2} * (xy - wz));
//    result[0, 2] = static_cast<T>(U{2} * (xz + wy));
//
//    result[1, 0] = static_cast<T>(U{2} * (xy + wz));
//    result[1, 1] = static_cast<T>(U{1} - U{2} * (xx + zz));
//    result[1, 2] = static_cast<T>(U{2} * (yz - wx));
//
//    result[2, 0] = static_cast<T>(U{2} * (xz - wy));
//    result[2, 1] = static_cast<T>(U{2} * (yz + wx));
//    result[2, 2] = static_cast<T>(U{1} - U{2} * (xx + yy));
//
//    return result;
//}
//
//template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
//template <float_num U, std::size_t R, std::size_t C>
//    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
//inline constexpr MatSIMD<T, RowCount, ColCount, Layout> MatSIMD<T, RowCount, ColCount, Layout>::TRS(const VecSIMD<U, 3>& translation,
//                                                                                        const QuaternionSIMD<U>& rotation,
//                                                                                        const VecSIMD<U, 3>& scale) {
//    return Translate(translation) * Rotate(rotation) * MatSIMD::Scale(scale);
//}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>
MatSIMD<T, RowCount, ColCount, Layout>::Perspective(U fovYRadians, U aspect, U nearPlane, U farPlane) {
    const U tanHalfFovY = std::tan(fovYRadians * U{0.5});

    MatSIMD result = Zero();

    result[0, 0] = static_cast<T>(U{1} / (aspect * tanHalfFovY));
    result[1, 1] = static_cast<T>(U{1} / tanHalfFovY);
    result[2, 2] = static_cast<T>(-(farPlane + nearPlane) / (farPlane - nearPlane));
    result[2, 3] = static_cast<T>(-(U{2} * farPlane * nearPlane) / (farPlane - nearPlane));
    result[3, 2] = static_cast<T>(-U{1});

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>
MatSIMD<T, RowCount, ColCount, Layout>::Ortho(U left, U right, U bottom, U top, U nearPlane, U farPlane) {
    MatSIMD result = Identity();

    result[0, 0] = static_cast<T>(U{2} / (right - left));
    result[1, 1] = static_cast<T>(U{2} / (top - bottom));
    result[2, 2] = static_cast<T>(-U{2} / (farPlane - nearPlane));
    result[0, 3] = static_cast<T>(-(right + left) / (right - left));
    result[1, 3] = static_cast<T>(-(top + bottom) / (top - bottom));
    result[2, 3] = static_cast<T>(-(farPlane + nearPlane) / (farPlane - nearPlane));

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr MatSIMD<T, RowCount, ColCount, Layout>
MatSIMD<T, RowCount, ColCount, Layout>::LookAt(const VecSIMD<U, 3>& eye, const VecSIMD<U, 3>& target, const VecSIMD<U, 3>& up) {
    const VecSIMD<U, 3> forward = (target - eye).Normalized();
    const VecSIMD<U, 3> right = Cross(forward, up).Normalized();
    const VecSIMD<U, 3> trueUp = Cross(right, forward);

    MatSIMD result = Identity();

    result[0, 0] = static_cast<T>(right[0]);
    result[0, 1] = static_cast<T>(right[1]);
    result[0, 2] = static_cast<T>(right[2]);

    result[1, 0] = static_cast<T>(trueUp[0]);
    result[1, 1] = static_cast<T>(trueUp[1]);
    result[1, 2] = static_cast<T>(trueUp[2]);

    result[2, 0] = static_cast<T>(-forward[0]);
    result[2, 1] = static_cast<T>(-forward[1]);
    result[2, 2] = static_cast<T>(-forward[2]);

    result[0, 3] = static_cast<T>(-Dot(right, eye));
    result[1, 3] = static_cast<T>(-Dot(trueUp, eye));
    result[2, 3] = static_cast<T>(Dot(forward, eye));

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr VecSIMD<std::common_type_t<T, U>, 3>
MatSIMD<T, RowCount, ColCount, Layout>::MultiplyPoint(const VecSIMD<U, 3>& point) const {
    using Result = std::common_type_t<T, U>;

    const Result x = static_cast<Result>((*this)[0, 0]) * static_cast<Result>(point[0]) +
                        static_cast<Result>((*this)[0, 1]) * static_cast<Result>(point[1]) +
                        static_cast<Result>((*this)[0, 2]) * static_cast<Result>(point[2]) +
                        static_cast<Result>((*this)[0, 3]);
    const Result y = static_cast<Result>((*this)[1, 0]) * static_cast<Result>(point[0]) +
                        static_cast<Result>((*this)[1, 1]) * static_cast<Result>(point[1]) +
                        static_cast<Result>((*this)[1, 2]) * static_cast<Result>(point[2]) +
                        static_cast<Result>((*this)[1, 3]);
    const Result z = static_cast<Result>((*this)[2, 0]) * static_cast<Result>(point[0]) +
                        static_cast<Result>((*this)[2, 1]) * static_cast<Result>(point[1]) +
                        static_cast<Result>((*this)[2, 2]) * static_cast<Result>(point[2]) +
                        static_cast<Result>((*this)[2, 3]);

    return VecSIMD<Result, 3>{x, y, z};
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <float_num U, std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr VecSIMD<std::common_type_t<T, U>, 3>
MatSIMD<T, RowCount, ColCount, Layout>::MultiplyVector(const VecSIMD<U, 3>& vector) const {
    using Result = std::common_type_t<T, U>;

    const Result x = static_cast<Result>((*this)[0, 0]) * static_cast<Result>(vector[0]) +
                        static_cast<Result>((*this)[0, 1]) * static_cast<Result>(vector[1]) +
                        static_cast<Result>((*this)[0, 2]) * static_cast<Result>(vector[2]);
    const Result y = static_cast<Result>((*this)[1, 0]) * static_cast<Result>(vector[0]) +
                        static_cast<Result>((*this)[1, 1]) * static_cast<Result>(vector[1]) +
                        static_cast<Result>((*this)[1, 2]) * static_cast<Result>(vector[2]);
    const Result z = static_cast<Result>((*this)[2, 0]) * static_cast<Result>(vector[0]) +
                        static_cast<Result>((*this)[2, 1]) * static_cast<Result>(vector[1]) +
                        static_cast<Result>((*this)[2, 2]) * static_cast<Result>(vector[2]);

    return VecSIMD<Result, 3>{x, y, z};
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr VecSIMD<T, 3> MatSIMD<T, RowCount, ColCount, Layout>::ExtractPosition() const {
    return VecSIMD<T, 3>{(*this)[0, 3], (*this)[1, 3], (*this)[2, 3]};
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
template <std::size_t R, std::size_t C>
    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
inline constexpr VecSIMD<T, 3> MatSIMD<T, RowCount, ColCount, Layout>::ExtractScale() const {
    const VecSIMD<T, 3> col0{(*this)[0, 0], (*this)[1, 0], (*this)[2, 0]};
    const VecSIMD<T, 3> col1{(*this)[0, 1], (*this)[1, 1], (*this)[2, 1]};
    const VecSIMD<T, 3> col2{(*this)[0, 2], (*this)[1, 2], (*this)[2, 2]};

    return VecSIMD<T, 3>{col0.Length(), col1.Length(), col2.Length()};
}

//template <float_num T, std::size_t RowCount, std::size_t ColCount, typename Layout>
//template <float_num U, std::size_t R, std::size_t C>
//    requires(R == RowCount && C == ColCount && R == 4 && C == 4)
//inline constexpr QuaternionSIMD<U> MatSIMD<T, RowCount, ColCount, Layout>::ExtractRotation() const {
//    const VecSIMD<T, 3> scale = ExtractScale();
//
//    const T sx = scale[0] != T{0} ? scale[0] : T{1};
//    const T sy = scale[1] != T{0} ? scale[1] : T{1};
//    const T sz = scale[2] != T{0} ? scale[2] : T{1};
//
//    const T m00 = (*this)[0, 0] / sx;
//    const T m10 = (*this)[1, 0] / sx;
//    const T m20 = (*this)[2, 0] / sx;
//    const T m01 = (*this)[0, 1] / sy;
//    const T m11 = (*this)[1, 1] / sy;
//    const T m21 = (*this)[2, 1] / sy;
//    const T m02 = (*this)[0, 2] / sz;
//    const T m12 = (*this)[1, 2] / sz;
//    const T m22 = (*this)[2, 2] / sz;
//
//    const T trace = m00 + m11 + m22;
//
//    if (trace > T{0}) {
//        const T s = std::sqrt(trace + T{1}) * T{2};
//        return QuaternionSIMD<T>((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, s * T{0.25});
//    }
//
//    if (m00 > m11 && m00 > m22) {
//        const T s = std::sqrt(T{1} + m00 - m11 - m22) * T{2};
//        return QuaternionSIMD<T>(s * T{0.25}, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s);
//    }
//
//    if (m11 > m22) {
//        const T s = std::sqrt(T{1} + m11 - m00 - m22) * T{2};
//        return QuaternionSIMD<T>((m01 + m10) / s, s * T{0.25}, (m12 + m21) / s, (m02 - m20) / s);
//    }
//
//    const T s = std::sqrt(T{1} + m22 - m00 - m11) * T{2};
//    return QuaternionSIMD<T>((m02 + m20) / s, (m12 + m21) / s, s * T{0.25}, (m10 - m01) / s);
//}

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator+(const MatSIMD<T, RowCount, ColCount, Layout>& a, const MatSIMD<U, RowCount, ColCount, Layout>& b)
    -> MatSIMDCommon<T, U, RowCount, ColCount, Layout> {
    using R = std::common_type_t<T, U>;

    MatSIMDCommon<T, U, RowCount, ColCount, Layout> result;

    for (std::size_t row = 0; row < RowCount; ++row) {
        for (std::size_t col = 0; col < ColCount; ++col) {
            result[row, col] = static_cast<R>(a[row, col]) + static_cast<R>(b[row, col]);
        }
    }

    return result;
}

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator-(const MatSIMD<T, RowCount, ColCount, Layout>& a, const MatSIMD<U, RowCount, ColCount, Layout>& b)
    -> MatSIMDCommon<T, U, RowCount, ColCount, Layout> {
    using R = std::common_type_t<T, U>;

    MatSIMDCommon<T, U, RowCount, ColCount, Layout> result;

    for (std::size_t row = 0; row < RowCount; ++row) {
        for (std::size_t col = 0; col < ColCount; ++col) {
            result[row, col] = static_cast<R>(a[row, col]) - static_cast<R>(b[row, col]);
        }
    }

    return result;
}

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator*(const MatSIMD<T, RowCount, ColCount, Layout>& MatSIMD, U scalar)
    -> MatSIMDCommon<T, U, RowCount, ColCount, Layout> {
    using R = std::common_type_t<T, U>;

    MatSIMDCommon<T, U, RowCount, ColCount, Layout> result;

    for (std::size_t row = 0; row < RowCount; ++row) {
        for (std::size_t col = 0; col < ColCount; ++col) {
            result[row, col] = static_cast<R>(MatSIMD[row, col]) * static_cast<R>(scalar);
        }
    }

    return result;
}

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator*(U scalar, const MatSIMD<T, RowCount, ColCount, Layout>& MatSIMD)
    -> MatSIMDCommon<T, U, RowCount, ColCount, Layout> {
    return MatSIMD * scalar;
}

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator/(const MatSIMD<T, RowCount, ColCount, Layout>& MatSIMD, U scalar)
    -> MatSIMDCommon<T, U, RowCount, ColCount, Layout> {
    using R = std::common_type_t<T, U>;

    MatSIMDCommon<T, U, RowCount, ColCount, Layout> result;

    for (std::size_t row = 0; row < RowCount; ++row) {
        for (std::size_t col = 0; col < ColCount; ++col) {
            result[row, col] = static_cast<R>(MatSIMD[row, col]) / static_cast<R>(scalar);
        }
    }

    return result;
}

template <float_num T, float_num U, std::size_t A, std::size_t B, std::size_t C, typename Layout>
inline constexpr auto operator*(const MatSIMD<T, A, B, Layout>& lhs, const MatSIMD<U, B, C, Layout>& rhs)
    -> MatSIMDCommon<T, U, A, C, Layout> {
    using R = std::common_type_t<T, U>;

    MatSIMDCommon<T, U, A, C, Layout> result;

    for (std::size_t row = 0; row < A; ++row) {
        for (std::size_t col = 0; col < C; ++col) {
            R value{};

            for (std::size_t i = 0; i < B; ++i) {
                value += static_cast<R>(lhs[row, i]) * static_cast<R>(rhs[i, col]);
            }

            result[row, col] = value;
        }
    }

    return result;
}

template <float_num T, std::size_t RowCount, std::size_t ColCount, typename FromLayout, typename ToLayout>
inline constexpr MatSIMD<T, RowCount, ColCount, ToLayout> MatCastLayout(const MatSIMD<T, RowCount, ColCount, FromLayout>& mat) {
    MatSIMD<T, RowCount, ColCount, ToLayout> result;

    for (std::size_t row = 0; row < RowCount; ++row) {
        for (std::size_t col = 0; col < ColCount; ++col) {
            result[row, col] = mat[row, col];
        }
    }

    return result;
}

template <float_num T, float_num U, std::size_t RowCount, std::size_t ColCount, typename Layout>
inline constexpr auto operator*(const MatSIMD<T, RowCount, ColCount, Layout>& mat, const VecSIMD<U, ColCount>& vec)
    -> VecSIMD<std::common_type_t<T, U>, RowCount> {
    using R = std::common_type_t<T, U>;

    VecSIMD<R, RowCount> result;

    for (std::size_t row = 0; row < RowCount; ++row) {
        R value{};

        for (std::size_t col = 0; col < ColCount; ++col) {
            value += static_cast<R>(mat[row, col]) * static_cast<R>(vec[col]);
        }

        result[row] = value;
    }

    return result;
}

template <float_num T>
inline constexpr T ToRadians(T degrees) {
    return degrees * (std::numbers::pi_v<T> / T{180});
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

        const VecSIMD<T, 3> col0{mat[0, 0], mat[1, 0], mat[2, 0]};
        const VecSIMD<T, 3> col1{mat[0, 1], mat[1, 1], mat[2, 1]};
        const VecSIMD<T, 3> col2{mat[0, 2], mat[1, 2], mat[2, 2]};

        if (col0.Length() <= T{} || col1.Length() <= T{} || col2.Length() <= T{}) {
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
