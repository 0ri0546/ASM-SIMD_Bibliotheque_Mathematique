#include "pch.h"
#include "CppUnitTest.h"

#include "MatSIMD.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace {
    constexpr float kPi = std::numbers::pi_v<float>;

    // Compares two floating-point values through Assert::AreEqual's tolerance overload.
    void ExpectNear(double expected, double actual, double tolerance = 1e-4) {
        Assert::AreEqual(expected, actual, tolerance);
    }

    template <float_num T>
    void ExpectNear(const VecSIMD<T, 3>& a, const VecSIMD<T, 3>& b, T tolerance = static_cast<T>(1e-4)) {
        ExpectNear(a[0], b[0], tolerance);
        ExpectNear(a[1], b[1], tolerance);
        ExpectNear(a[2], b[2], tolerance);
    }
} // namespace

namespace MathLibraryTests
{
    TEST_CLASS(MatSIMDTests)
    {
    public:
        // =========================================================================
        // Construction
        // =========================================================================

        TEST_METHOD(DefaultConstruction) {
            const MatSIMD<float, 2, 3> mat;

            for (std::size_t row = 0; row < 2; ++row) {
                for (std::size_t col = 0; col < 3; ++col) {
                    Assert::AreEqual(0.0f, (mat[row, col]));
                }
            }
        }

        TEST_METHOD(ValueConstruction) {
            const MatSIMD<float, 2, 3> mat{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };

            Assert::AreEqual(1.0f, (mat[0, 0]));
            Assert::AreEqual(2.0f, (mat[0, 1]));
            Assert::AreEqual(3.0f, (mat[0, 2]));

            Assert::AreEqual(4.0f, (mat[1, 0]));
            Assert::AreEqual(5.0f, (mat[1, 1]));
            Assert::AreEqual(6.0f, (mat[1, 2]));
        }

        TEST_METHOD(ValueConstructionWithConversion) {
            const MatSIMD<float, 2, 2> mat{ 1.0f, 2.0f, 3.0f, 4.0f };

            Assert::AreEqual(1.0f, (mat[0, 0]));
            Assert::AreEqual(2.0f, (mat[0, 1]));
            Assert::AreEqual(3.0f, (mat[1, 0]));
            Assert::AreEqual(4.0f, (mat[1, 1]));
        }

        // =========================================================================
        // Access
        // =========================================================================

        TEST_METHOD(BracketAccess) {
            MatSIMD<float, 3, 3> mat{};

            mat[1, 2] = 42.0f;

            Assert::AreEqual(42.0f, (mat[1, 2]));
        }

        TEST_METHOD(ParenthesisAccess) {
            MatSIMD<float, 3, 3> mat{};

            mat(2, 1) = 42.0f;

            Assert::AreEqual(42.0f, (mat(2, 1)));
        }

        TEST_METHOD(ConstAccess) {
            const MatSIMD<float, 2, 2> mat{ 1.0f, 2.0f, 3.0f, 4.0f };

            Assert::AreEqual(1.0f, (mat[0, 0]));
            Assert::AreEqual(4.0f, (mat[1, 1]));
        }

        TEST_METHOD(Data) {
            MatSIMD<float, 2, 3> mat{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };

            Assert::IsNotNull(mat.Data());

            mat.Data()[0] = 42.0f;

            Assert::AreEqual(42.0f, (mat[0, 0]));
        }

        // =========================================================================
        // Dimensions
        // =========================================================================

        TEST_METHOD(Dimensions) {
            using Matrix = MatSIMD<float, 3, 4>;

            Assert::AreEqual(static_cast<std::size_t>(3), Matrix::Rows());
            Assert::AreEqual(static_cast<std::size_t>(4), Matrix::Cols());
            Assert::AreEqual(static_cast<std::size_t>(12), Matrix::Size());
        }

        TEST_METHOD(DimensionsAreCompileTimeConstants) {
            using Matrix = MatSIMD<float, 3, 4>;

            static_assert(Matrix::Rows() == 3);
            static_assert(Matrix::Cols() == 4);
            static_assert(Matrix::Size() == 12);
        }

        // =========================================================================
        // MDSpan
        // =========================================================================

        TEST_METHOD(MDSpan) {
            MatSIMD<float, 2, 3> mat{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };

            const auto span = mat.MDSpan();

            Assert::AreEqual(static_cast<std::size_t>(2), span.extent(0));
            Assert::AreEqual(static_cast<std::size_t>(3), span.extent(1));

            Assert::AreEqual(1.0f, (span[0, 0]));
            Assert::AreEqual(3.0f, (span[0, 2]));
            Assert::AreEqual(4.0f, (span[1, 0]));
            Assert::AreEqual(6.0f, (span[1, 2]));

            span[1, 1] = 42.0f;

            Assert::AreEqual(42.0f, (mat[1, 1]));
        }

        TEST_METHOD(ConstMDSpan) {
            const MatSIMD<float, 2, 2> mat{ 1.0f, 2.0f, 3.0f, 4.0f };

            const auto span = mat.MDSpan();

            Assert::AreEqual(static_cast<std::size_t>(2), span.extent(0));
            Assert::AreEqual(static_cast<std::size_t>(2), span.extent(1));

            Assert::AreEqual(1.0f, (span[0, 0]));
            Assert::AreEqual(4.0f, (span[1, 1]));
        }

        // =========================================================================
        // Equality
        // =========================================================================

        TEST_METHOD(Equality) {
            const MatSIMD<float, 2, 2> a{ 1, 2, 3, 4 };

            const MatSIMD<float, 2, 2> b{ 1, 2, 3, 4 };

            const MatSIMD<float, 2, 2> c{ 1, 2, 3, 5 };

            Assert::IsTrue(a == b);
            Assert::IsFalse(a == c);
        }

        TEST_METHOD(EqualityDifferentNumericTypes) {
            const MatSIMD<float, 2, 2> a{ 1, 2, 3, 4 };

            const MatSIMD<double, 2, 2> b{ 1.0, 2.0, 3.0, 4.0 };

            Assert::IsTrue(a == b);
        }

        TEST_METHOD(EqualityDifferentLayouts) {
            const MatSIMDRowMajor<float, 2, 2> row{ 1, 2, 3, 4 };

            const MatSIMDColumnMajor<float, 2, 2> column{ 1, 2, 3, 4 };

            Assert::IsTrue(row == column);
        }

        // =========================================================================
        // Addition
        // =========================================================================

        TEST_METHOD(Addition) {
            MatSIMD<float, 2, 2> a{ 1, 2, 3, 4 };

            MatSIMD<float, 2, 2> b{ 5, 6, 7, 8 };

            const auto result = a + b;

            static_assert(std::is_same_v<decltype(result), const MatSIMD<float, 2, 2>>);

            Assert::AreEqual(6.0f, (result[0, 0]));
            Assert::AreEqual(8.0f, (result[0, 1]));
            Assert::AreEqual(10.0f, (result[1, 0]));
            Assert::AreEqual(12.0f, (result[1, 1]));
        }

        TEST_METHOD(AdditionDifferentNumericTypes) {
            const MatSIMD<float, 2, 2> a{ 1, 2, 3, 4 };

            const MatSIMD<double, 2, 2> b{ 0.5, 1.5, 2.5, 3.5 };

            const auto result = a + b;

            static_assert(std::is_same_v<decltype(result), const MatSIMD<double, 2, 2>>);

            ExpectNear(1.5, (result[0, 0]));
            ExpectNear(3.5, (result[0, 1]));
            ExpectNear(5.5, (result[1, 0]));
            ExpectNear(7.5, (result[1, 1]));
        }

        TEST_METHOD(AdditionAssignment) {
            MatSIMD<float, 2, 2> a{ 1, 2, 3, 4 };

            const MatSIMD<float, 2, 2> b{ 5, 6, 7, 8 };

            a += b;

            Assert::AreEqual(6.0f, (a[0, 0]));
            Assert::AreEqual(8.0f, (a[0, 1]));
            Assert::AreEqual(10.0f, (a[1, 0]));
            Assert::AreEqual(12.0f, (a[1, 1]));
        }

        // =========================================================================
        // Subtraction
        // =========================================================================

        TEST_METHOD(Subtraction) {
            const MatSIMD<float, 2, 2> a{ 5, 6, 7, 8 };

            const MatSIMD<float, 2, 2> b{ 1, 2, 3, 4 };

            const auto result = a - b;

            Assert::AreEqual(4.0f, (result[0, 0]));
            Assert::AreEqual(4.0f, (result[0, 1]));
            Assert::AreEqual(4.0f, (result[1, 0]));
            Assert::AreEqual(4.0f, (result[1, 1]));
        }

        TEST_METHOD(SubtractionAssignment) {
            MatSIMD<float, 2, 2> a{ 5, 6, 7, 8 };

            const MatSIMD<float, 2, 2> b{ 1, 2, 3, 4 };

            a -= b;

            Assert::AreEqual(4.0f, (a[0, 0]));
            Assert::AreEqual(4.0f, (a[0, 1]));
            Assert::AreEqual(4.0f, (a[1, 0]));
            Assert::AreEqual(4.0f, (a[1, 1]));
        }

        // =========================================================================
        // Unary operators
        // =========================================================================

        TEST_METHOD(UnaryPlus) {
            const MatSIMD<float, 2, 2> a{ 1, 2, 3, 4 };

            const auto result = +a;

            Assert::IsTrue(result == a);
        }

        TEST_METHOD(UnaryMinus) {
            const MatSIMD<float, 2, 2> a{ 1, -2, 3, -4 };

            const auto result = -a;

            Assert::AreEqual(-1.0f, (result[0, 0]));
            Assert::AreEqual(2.0f, (result[0, 1]));
            Assert::AreEqual(-3.0f, (result[1, 0]));
            Assert::AreEqual(4.0f, (result[1, 1]));
        }

        // =========================================================================
        // Scalar multiplication
        // =========================================================================

        TEST_METHOD(ScalarMultiplication) {
            const MatSIMD<float, 2, 2> mat{ 1, 2, 3, 4 };

            const auto result = mat * 2.0f;

            Assert::AreEqual(2.0f, (result[0, 0]));
            Assert::AreEqual(4.0f, (result[0, 1]));
            Assert::AreEqual(6.0f, (result[1, 0]));
            Assert::AreEqual(8.0f, (result[1, 1]));
        }

        TEST_METHOD(ScalarMultiplicationReversed) {
            const MatSIMD<float, 2, 2> mat{ 1, 2, 3, 4 };

            const auto result = 2.0f * mat;

            Assert::AreEqual(2.0f, (result[0, 0]));
            Assert::AreEqual(4.0f, (result[0, 1]));
            Assert::AreEqual(6.0f, (result[1, 0]));
            Assert::AreEqual(8.0f, (result[1, 1]));
        }

        TEST_METHOD(ScalarMultiplicationDifferentType) {
            const MatSIMD<float, 2, 2> mat{ 1, 2, 3, 4 };

            const auto result = mat * 0.5;

            static_assert(std::is_same_v<decltype(result), const MatSIMD<double, 2, 2>>);

            ExpectNear(0.5, (result[0, 0]));
            ExpectNear(1.0, (result[0, 1]));
            ExpectNear(1.5, (result[1, 0]));
            ExpectNear(2.0, (result[1, 1]));
        }

        TEST_METHOD(ScalarMultiplicationAssignment) {
            MatSIMD<float, 2, 2> mat{ 1, 2, 3, 4 };

            mat *= 2.0f;

            Assert::AreEqual(2.0f, (mat[0, 0]));
            Assert::AreEqual(4.0f, (mat[0, 1]));
            Assert::AreEqual(6.0f, (mat[1, 0]));
            Assert::AreEqual(8.0f, (mat[1, 1]));
        }

        // =========================================================================
        // Scalar division
        // =========================================================================

        TEST_METHOD(ScalarDivision) {
            const MatSIMD<float, 2, 2> mat{ 2, 4, 6, 8 };

            const auto result = mat / 2.0f;

            ExpectNear(1.0f, (result[0, 0]));
            ExpectNear(2.0f, (result[0, 1]));
            ExpectNear(3.0f, (result[1, 0]));
            ExpectNear(4.0f, (result[1, 1]));
        }

        TEST_METHOD(ScalarDivisionDifferentType) {
            const MatSIMD<float, 2, 2> mat{ 2, 4, 6, 8 };

            const auto result = mat / 2.0;

            static_assert(std::is_same_v<decltype(result), const MatSIMD<double, 2, 2>>);

            ExpectNear(1.0, (result[0, 0]));
            ExpectNear(2.0, (result[0, 1]));
            ExpectNear(3.0, (result[1, 0]));
            ExpectNear(4.0, (result[1, 1]));
        }

        TEST_METHOD(ScalarDivisionAssignment) {
            MatSIMD<float, 2, 2> mat{ 2, 4, 6, 8 };

            mat /= 2.0f;

            ExpectNear(1.0f, (mat[0, 0]));
            ExpectNear(2.0f, (mat[0, 1]));
            ExpectNear(3.0f, (mat[1, 0]));
            ExpectNear(4.0f, (mat[1, 1]));
        }

        // =========================================================================
        // Matrix multiplication
        // =========================================================================

        TEST_METHOD(MatrixMultiplication) {
            const MatSIMD<float, 2, 3> a{ 1, 2, 3, 4, 5, 6 };

            const MatSIMD<float, 3, 2> b{ 7, 8, 9, 10, 11, 12 };

            const auto result = a * b;

            static_assert(std::is_same_v<decltype(result), const MatSIMD<float, 2, 2>>);

            Assert::AreEqual(58.0f, (result[0, 0]));
            Assert::AreEqual(64.0f, (result[0, 1]));
            Assert::AreEqual(139.0f, (result[1, 0]));
            Assert::AreEqual(154.0f, (result[1, 1]));
        }

        TEST_METHOD(MatrixMultiplicationDifferentNumericTypes) {
            const MatSIMD<float, 2, 3> a{ 1, 2, 3, 4, 5, 6 };

            const MatSIMD<double, 3, 2> b{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

            const auto result = a * b;

            static_assert(std::is_same_v<decltype(result), const MatSIMD<double, 2, 2>>);

            ExpectNear(22.0, (result[0, 0]));
            ExpectNear(28.0, (result[0, 1]));
            ExpectNear(49.0, (result[1, 0]));
            ExpectNear(64.0, (result[1, 1]));
        }

        TEST_METHOD(SquareMatrixMultiplication) {
            const MatSIMD<float, 2, 2> a{ 1, 2, 3, 4 };

            const MatSIMD<float, 2, 2> b{ 5, 6, 7, 8 };

            const auto result = a * b;

            Assert::AreEqual(19.0f, (result[0, 0]));
            Assert::AreEqual(22.0f, (result[0, 1]));
            Assert::AreEqual(43.0f, (result[1, 0]));
            Assert::AreEqual(50.0f, (result[1, 1]));
        }

        // =========================================================================
        // Transpose
        // =========================================================================

        TEST_METHOD(Transpose) {
            const MatSIMD<float, 2, 3> mat{ 1, 2, 3, 4, 5, 6 };

            const auto result = mat.Transpose();

            static_assert(std::is_same_v<decltype(result), const MatSIMD<float, 3, 2>>);

            Assert::AreEqual(1.0f, (result[0, 0]));
            Assert::AreEqual(4.0f, (result[0, 1]));

            Assert::AreEqual(2.0f, (result[1, 0]));
            Assert::AreEqual(5.0f, (result[1, 1]));

            Assert::AreEqual(3.0f, (result[2, 0]));
            Assert::AreEqual(6.0f, (result[2, 1]));
        }

        TEST_METHOD(SquareTranspose) {
            const MatSIMD<float, 3, 3> mat{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            const auto result = mat.Transpose();

            Assert::AreEqual(1.0f, (result[0, 0]));
            Assert::AreEqual(4.0f, (result[0, 1]));
            Assert::AreEqual(7.0f, (result[0, 2]));

            Assert::AreEqual(2.0f, (result[1, 0]));
            Assert::AreEqual(5.0f, (result[1, 1]));
            Assert::AreEqual(8.0f, (result[1, 2]));

            Assert::AreEqual(3.0f, (result[2, 0]));
            Assert::AreEqual(6.0f, (result[2, 1]));
            Assert::AreEqual(9.0f, (result[2, 2]));
        }

        // =========================================================================
        // Identity
        // =========================================================================

        TEST_METHOD(Identity2x2) {
            const auto identity = MatSIMD<float, 2, 2>::Identity();

            Assert::AreEqual(1.0f, (identity[0, 0]));
            Assert::AreEqual(0.0f, (identity[0, 1]));
            Assert::AreEqual(0.0f, (identity[1, 0]));
            Assert::AreEqual(1.0f, (identity[1, 1]));
        }

        TEST_METHOD(Identity4x4) {
            const auto identity = MatSIMD<float, 4, 4>::Identity();

            for (std::size_t row = 0; row < 4; ++row) {
                for (std::size_t col = 0; col < 4; ++col) {
                    if (row == col) {
                        ExpectNear(1.0f, (identity[row, col]));
                    }
                    else {
                        ExpectNear(0.0f, (identity[row, col]));
                    }
                }
            }
        }

        TEST_METHOD(IdentityIsNeutralForMultiplication) {
            const MatSIMD<float, 3, 3> mat{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            const auto identity = MatSIMD<float, 3, 3>::Identity();

            Assert::IsTrue(mat * identity == mat);
            Assert::IsTrue(identity * mat == mat);
        }

        // =========================================================================
        // Determinant
        // =========================================================================

        TEST_METHOD(Determinant1x1) {
            const MatSIMD<float, 1, 1> mat{ 5.0f };

            ExpectNear(5.0f, mat.Determinant());
        }

        TEST_METHOD(Determinant2x2) {
            const MatSIMD<float, 2, 2> mat{ 1, 2, 3, 4 };

            ExpectNear(-2.0f, mat.Determinant());
        }

        TEST_METHOD(Determinant3x3) {
            const MatSIMD<float, 3, 3> mat{ 6, 1, 1, 4, -2, 5, 2, 8, 7 };

            ExpectNear(-306.0f, mat.Determinant());
        }

        TEST_METHOD(DeterminantIdentity) {
            const auto identity = MatSIMD<float, 4, 4>::Identity();

            ExpectNear(1.0f, identity.Determinant());
        }

        TEST_METHOD(DeterminantSingular) {
            const MatSIMD<float, 2, 2> mat{ 1, 2, 2, 4 };

            ExpectNear(0.0f, mat.Determinant());
        }

        // =========================================================================
        // Inverse
        // =========================================================================

        TEST_METHOD(Inverse2x2) {
            const MatSIMD<float, 2, 2> mat{ 4.0, 7.0, 2.0, 6.0 };

            const auto inverse = mat.Inverse();

            ExpectNear(0.6, (inverse[0, 0]), 1e-5);
            ExpectNear(-0.7, (inverse[0, 1]), 1e-5);
            ExpectNear(-0.2, (inverse[1, 0]), 1e-5);
            ExpectNear(0.4, (inverse[1, 1]), 1e-5);
        }

        TEST_METHOD(InverseIdentity) {
            const auto identity = MatSIMD<float, 4, 4>::Identity();

            const auto inverse = identity.Inverse();

            Assert::IsTrue(inverse == identity);
        }

        TEST_METHOD(MatrixTimesInverseIsIdentity) {
            const MatSIMD<float, 2, 2> mat{ 4.0, 7.0, 2.0, 6.0 };

            const auto inverse = mat.Inverse();
            const auto result = mat * inverse;

            const auto identity = MatSIMD<float, 2, 2>::Identity();

            for (std::size_t row = 0; row < 2; ++row) {
                for (std::size_t col = 0; col < 2; ++col) {
                    ExpectNear((identity[row, col]), (result[row, col]), 1e-5);
                }
            }
        }

        TEST_METHOD(SingularMatrixInverse) {
            const MatSIMD<float, 2, 2> mat{ 1.0, 2.0, 2.0, 4.0 };

            const auto inverse = mat.Inverse();
            const auto b = MatSIMD<float, 2, 2>{};

            Assert::IsTrue(inverse == b);
        }

        // =========================================================================
        // Zero
        // =========================================================================

        TEST_METHOD(Zero) {
            const auto mat = MatSIMD<float, 4, 4>::Zero();

            for (std::size_t row = 0; row < 4; ++row) {
                for (std::size_t col = 0; col < 4; ++col) {
                    ExpectNear(0.0f, (mat[row, col]));
                }
            }
        }

        // =========================================================================
        // Rotation factories
        // =========================================================================

        TEST_METHOD(RotationXRotatesYIntoZ) {
            const MatSIMD<float, 4, 4> rot = MatSIMD<float, 4, 4>::RotationX(kPi / 2.0f);
            constexpr VecSIMD<float, 3> v{ 0.0f, 1.0f, 0.0f };

            const VecSIMD<float, 3> result = rot.MultiplyVector(v);

            ExpectNear(result, VecSIMD<float, 3>{ 0.0f, 0.0f, 1.0f });
        }

        TEST_METHOD(RotationYRotatesZIntoX) {
            const MatSIMD<float, 4, 4> rot = MatSIMD<float, 4, 4>::RotationY(kPi / 2.0f);
            constexpr VecSIMD<float, 3> v{ 0.0f, 0.0f, 1.0f };

            const VecSIMD<float, 3> result = rot.MultiplyVector(v);

            ExpectNear(result, VecSIMD<float, 3>{ 1.0f, 0.0f, 0.0f });
        }

        TEST_METHOD(RotationZRotatesXIntoY) {
            const MatSIMD<float, 4, 4> rot = MatSIMD<float, 4, 4>::RotationZ(kPi / 2.0f);
            constexpr VecSIMD<float, 3> v{ 1.0f, 0.0f, 0.0f };

            const VecSIMD<float, 3> result = rot.MultiplyVector(v);

            ExpectNear(result, VecSIMD<float, 3>{ 0.0f, 1.0f, 0.0f });
        }

        TEST_METHOD(RotationXByZeroIsIdentity) {
            const MatSIMD<float, 4, 4> rot = MatSIMD<float, 4, 4>::RotationX(0.0f);
            const MatSIMD<float, 4, 4> identity = MatSIMD<float, 4, 4>::Identity();

            for (std::size_t row = 0; row < 4; ++row) {
                for (std::size_t col = 0; col < 4; ++col) {
                    ExpectNear((identity[row, col]), (rot[row, col]), 1e-6f);
                }
            }
        }

        // =========================================================================
        // Scale / Translate factories
        // =========================================================================

        TEST_METHOD(ScaleFactoryScalesPoint) {
            const MatSIMD<float, 4, 4> mat = MatSIMD<float, 4, 4>::Scale(VecSIMD<float, 3>{ 2.0f, 3.0f, 4.0f });
            constexpr VecSIMD<float, 3> point{ 1.0f, 1.0f, 1.0f };

            const VecSIMD<float, 3> result = mat.MultiplyPoint(point);

            ExpectNear(result, VecSIMD<float, 3>{ 2.0f, 3.0f, 4.0f });
        }

        TEST_METHOD(TranslateFactoryTranslatesPoint) {
            const MatSIMD<float, 4, 4> mat = MatSIMD<float, 4, 4>::Translate(VecSIMD<float, 3>{ 1.0f, 2.0f, 3.0f });
            constexpr VecSIMD<float, 3> point{ 0.0f, 0.0f, 0.0f };

            const VecSIMD<float, 3> result = mat.MultiplyPoint(point);

            ExpectNear(result, VecSIMD<float, 3>{ 1.0f, 2.0f, 3.0f });
        }

        TEST_METHOD(TranslateFactoryDoesNotAffectVector) {
            const MatSIMD<float, 4, 4> mat = MatSIMD<float, 4, 4>::Translate(VecSIMD<float, 3>{ 1.0f, 2.0f, 3.0f });
            constexpr VecSIMD<float, 3> direction{ 5.0f, 0.0f, 0.0f };

            const VecSIMD<float, 3> result = mat.MultiplyVector(direction);

            ExpectNear(result, direction);
        }

        // =========================================================================
        // Rotate (from quaternion)
        // =========================================================================

        TEST_METHOD(RotateMatchesQuaternionRotateVector) {
            const QuaternionSIMDf rotation = QuaternionSIMDf::FromAxisAngle(VecSIMD<float, 3>{ 0.0f, 1.0f, 0.0f }, kPi / 3.0f);
            const MatSIMD<float, 4, 4> mat = MatSIMD<float, 4, 4>::Rotate(rotation);

            constexpr VecSIMD<float, 3> v{ 1.0f, 0.0f, 0.0f };

            ExpectNear(mat.MultiplyVector(v), rotation.RotateVector(v));
        }

        TEST_METHOD(RotateIdentityQuaternionIsIdentityMatrix) {
            const MatSIMD<float, 4, 4> mat = MatSIMD<float, 4, 4>::Rotate(QuaternionSIMDf::Identity());
            const MatSIMD<float, 4, 4> identity = MatSIMD<float, 4, 4>::Identity();

            for (std::size_t row = 0; row < 4; ++row) {
                for (std::size_t col = 0; col < 4; ++col) {
                    ExpectNear((identity[row, col]), (mat[row, col]), 1e-6f);
                }
            }
        }

        // =========================================================================
        // TRS
        // =========================================================================

        TEST_METHOD(TRSAppliesScaleRotateTranslateInOrder) {
            constexpr VecSIMD<float, 3> translation{ 10.0f, 0.0f, 0.0f };
            const QuaternionSIMDf rotation = QuaternionSIMDf::Identity();
            constexpr VecSIMD<float, 3> scale{ 2.0f, 1.0f, 1.0f };

            const MatSIMD<float, 4, 4> trs = MatSIMD<float, 4, 4>::TRS(translation, rotation, scale);
            constexpr VecSIMD<float, 3> localPoint{ 1.0f, 0.0f, 0.0f };

            const VecSIMD<float, 3> worldPoint = trs.MultiplyPoint(localPoint);

            ExpectNear(worldPoint, VecSIMD<float, 3>{ 12.0f, 0.0f, 0.0f });
        }

        TEST_METHOD(TRSWithIdentityComponentsIsIdentity) {
            const MatSIMD<float, 4, 4> trs = MatSIMD<float, 4, 4>::TRS(VecSIMD<float, 3>{ 0.0f, 0.0f, 0.0f }, QuaternionSIMDf::Identity(), VecSIMD<float, 3>{ 1.0f, 1.0f, 1.0f });
            const MatSIMD<float, 4, 4> identity = MatSIMD<float, 4, 4>::Identity();

            for (std::size_t row = 0; row < 4; ++row) {
                for (std::size_t col = 0; col < 4; ++col) {
                    ExpectNear((identity[row, col]), (trs[row, col]), 1e-6f);
                }
            }
        }

        // =========================================================================
        // Perspective / Ortho / LookAt
        // =========================================================================

        TEST_METHOD(PerspectiveMapsNearPlaneToNegativeOneNDC) {
            const MatSIMD<float, 4, 4> persp = MatSIMD<float, 4, 4>::Perspective(kPi / 2.0f, 1.0f, 0.1f, 100.0f);

            const VecSIMD<float, 4> nearPoint{ 0.0f, 0.0f, -0.1f, 1.0f };
            const VecSIMD<float, 4> clip = persp * nearPoint;

            ExpectNear(-1.0f, clip[2] / clip[3], 1e-4f);
        }

        TEST_METHOD(PerspectiveMapsFarPlaneToPositiveOneNDC) {
            const MatSIMD<float, 4, 4> persp = MatSIMD<float, 4, 4>::Perspective(kPi / 2.0f, 1.0f, 0.1f, 100.0f);

            const VecSIMD<float, 4> farPoint{ 0.0f, 0.0f, -100.0f, 1.0f };
            const VecSIMD<float, 4> clip = persp * farPoint;

            ExpectNear(1.0f, clip[2] / clip[3], 1e-4f);
        }

        TEST_METHOD(OrthoMapsNearPlaneToNegativeOne) {
            const MatSIMD<float, 4, 4> ortho = MatSIMD<float, 4, 4>::Ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);

            const VecSIMD<float, 3> nearPoint{ 0.0f, 0.0f, -0.1f };
            const VecSIMD<float, 3> mapped = ortho.MultiplyPoint(nearPoint);

            ExpectNear(-1.0f, mapped[2], 1e-4f);
        }

        TEST_METHOD(OrthoMapsFarPlaneToPositiveOne) {
            const MatSIMD<float, 4, 4> ortho = MatSIMD<float, 4, 4>::Ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);

            const VecSIMD<float, 3> farPoint{ 0.0f, 0.0f, -100.0f };
            const VecSIMD<float, 3> mapped = ortho.MultiplyPoint(farPoint);

            ExpectNear(1.0f, mapped[2], 1e-4f);
        }

        TEST_METHOD(OrthoMapsSideBoundsToUnitRange) {
            const MatSIMD<float, 4, 4> ortho = MatSIMD<float, 4, 4>::Ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);

            const VecSIMD<float, 3> corner{ 1.0f, 1.0f, -0.1f };
            const VecSIMD<float, 3> mapped = ortho.MultiplyPoint(corner);

            ExpectNear(1.0f, mapped[0], 1e-4f);
            ExpectNear(1.0f, mapped[1], 1e-4f);
        }

        TEST_METHOD(LookAtMapsEyeToOrigin) {
            constexpr VecSIMD<float, 3> eye{ 0.0f, 0.0f, 5.0f };
            constexpr VecSIMD<float, 3> target{ 0.0f, 0.0f, 0.0f };
            constexpr VecSIMD<float, 3> up{ 0.0f, 1.0f, 0.0f };

            const MatSIMD<float, 4, 4> lookAt = MatSIMD<float, 4, 4>::LookAt(eye, target, up);
            const VecSIMD<float, 3> eyeInView = lookAt.MultiplyPoint(eye);

            ExpectNear(eyeInView, VecSIMD<float, 3>{ 0.0f, 0.0f, 0.0f });
        }

        TEST_METHOD(LookAtRotationIsOrthonormal) {
            constexpr VecSIMD<float, 3> eye{ 0.0f, 0.0f, 5.0f };
            constexpr VecSIMD<float, 3> target{ 0.0f, 0.0f, 0.0f };
            constexpr VecSIMD<float, 3> up{ 0.0f, 1.0f, 0.0f };

            const MatSIMD<float, 4, 4> lookAt = MatSIMD<float, 4, 4>::LookAt(eye, target, up);

            const VecSIMD<float, 3> r0{ lookAt[0, 0], lookAt[0, 1], lookAt[0, 2] };
            const VecSIMD<float, 3> r1{ lookAt[1, 0], lookAt[1, 1], lookAt[1, 2] };
            const VecSIMD<float, 3> r2{ lookAt[2, 0], lookAt[2, 1], lookAt[2, 2] };

            ExpectNear(1.0f, r0.Length(), 1e-4f);
            ExpectNear(1.0f, r1.Length(), 1e-4f);
            ExpectNear(1.0f, r2.Length(), 1e-4f);

            ExpectNear(0.0f, Dot(r0, r1), 1e-4f);
            ExpectNear(0.0f, Dot(r0, r2), 1e-4f);
            ExpectNear(0.0f, Dot(r1, r2), 1e-4f);
        }

        // =========================================================================
        // MultiplyPoint / MultiplyVector
        // =========================================================================

        TEST_METHOD(MultiplyPointAppliesTranslation) {
            const MatSIMD<float, 4, 4> mat = MatSIMD<float, 4, 4>::Translate(VecSIMD<float, 3>{ 1.0f, 2.0f, 3.0f });
            constexpr VecSIMD<float, 3> point{ 1.0f, 1.0f, 1.0f };

            const VecSIMD<float, 3> result = mat.MultiplyPoint(point);

            ExpectNear(result, VecSIMD<float, 3>{ 2.0f, 3.0f, 4.0f });
        }

        TEST_METHOD(MultiplyVectorIgnoresTranslation) {
            const MatSIMD<float, 4, 4> mat = MatSIMD<float, 4, 4>::Translate(VecSIMD<float, 3>{ 1.0f, 2.0f, 3.0f });
            constexpr VecSIMD<float, 3> vector{ 1.0f, 1.0f, 1.0f };

            const VecSIMD<float, 3> result = mat.MultiplyVector(vector);

            ExpectNear(result, vector);
        }

        // =========================================================================
        // Extract Position / Scale / Rotation
        // =========================================================================

        TEST_METHOD(ExtractPositionFromTRS) {
            constexpr VecSIMD<float, 3> translation{ 1.0f, 2.0f, 3.0f };
            const MatSIMD<float, 4, 4> trs = MatSIMD<float, 4, 4>::TRS(translation, QuaternionSIMDf::Identity(), VecSIMD<float, 3>{ 1.0f, 1.0f, 1.0f });

            ExpectNear(trs.ExtractPosition(), translation);
        }

        TEST_METHOD(ExtractScaleFromTRS) {
            constexpr VecSIMD<float, 3> scale{ 2.0f, 3.0f, 4.0f };
            const MatSIMD<float, 4, 4> trs = MatSIMD<float, 4, 4>::TRS(VecSIMD<float, 3>{ 0.0f, 0.0f, 0.0f }, QuaternionSIMDf::Identity(), scale);

            ExpectNear(trs.ExtractScale(), scale);
        }

        TEST_METHOD(ExtractRotationFromTRS) {
            const QuaternionSIMDf rotation = QuaternionSIMDf::FromAxisAngle(VecSIMD<float, 3>{ 0.0f, 1.0f, 0.0f }, kPi / 4.0f);
            const MatSIMD<float, 4, 4> trs = MatSIMD<float, 4, 4>::TRS(VecSIMD<float, 3>{ 0.0f, 0.0f, 0.0f }, rotation, VecSIMD<float, 3>{ 1.0f, 1.0f, 1.0f });

            const QuaternionSIMDf extracted = trs.ExtractRotation();

            ExpectNear(rotation.X(), extracted.X(), 1e-4f);
            ExpectNear(rotation.Y(), extracted.Y(), 1e-4f);
            ExpectNear(rotation.Z(), extracted.Z(), 1e-4f);
            ExpectNear(rotation.W(), extracted.W(), 1e-4f);
        }

        TEST_METHOD(ExtractRoundTripsFullTRS) {
            constexpr VecSIMD<float, 3> translation{ 5.0f, -2.0f, 1.0f };
            const QuaternionSIMDf rotation = QuaternionSIMDf::FromAxisAngle(VecSIMD<float, 3>{ 1.0f, 0.0f, 0.0f }, kPi / 6.0f);
            constexpr VecSIMD<float, 3> scale{ 2.0f, 1.0f, 3.0f };

            const MatSIMD<float, 4, 4> trs = MatSIMD<float, 4, 4>::TRS(translation, rotation, scale);

            ExpectNear(trs.ExtractPosition(), translation);
            ExpectNear(trs.ExtractScale(), scale);
        }

        // =========================================================================
        // Matrix-vector multiplication
        // =========================================================================

        TEST_METHOD(IdentityTimesVectorIsUnchanged) {
            const MatSIMD<float, 4, 4> identity = MatSIMD<float, 4, 4>::Identity();
            constexpr VecSIMD<float, 4> v{ 1.0f, 2.0f, 3.0f, 1.0f };

            const VecSIMD<float, 4> result = identity * v;

            ExpectNear(1.0f, result[0]);
            ExpectNear(2.0f, result[1]);
            ExpectNear(3.0f, result[2]);
            ExpectNear(1.0f, result[3]);
        }

        TEST_METHOD(MatrixTimesVectorAppliesTransform) {
            const MatSIMD<float, 4, 4> translate = MatSIMD<float, 4, 4>::Translate(VecSIMD<float, 3>{ 1.0f, 2.0f, 3.0f });
            constexpr VecSIMD<float, 4> v{ 0.0f, 0.0f, 0.0f, 1.0f };

            const VecSIMD<float, 4> result = translate * v;

            ExpectNear(1.0f, result[0]);
            ExpectNear(2.0f, result[1]);
            ExpectNear(3.0f, result[2]);
            ExpectNear(1.0f, result[3]);
        }

        // =========================================================================
        // ToRadiansSIMD
        // =========================================================================

        TEST_METHOD(ToRadiansSIMDZero) {
            ExpectNear(0.0f, ToRadiansSIMD(0.0f));
        }

        TEST_METHOD(ToRadiansSIMDOneEighty) {
            ExpectNear(kPi, ToRadiansSIMD(180.0f));
        }

        TEST_METHOD(ToRadiansSIMDNinety) {
            ExpectNear(kPi / 2.0f, ToRadiansSIMD(90.0f), 1e-5f);
        }

        // =========================================================================
        // ValidTRS
        // =========================================================================

        TEST_METHOD(ValidTRSAcceptsIdentity) {
            Assert::IsTrue(ValidTRS(MatSIMD<float, 4, 4>::Identity()));
        }

        TEST_METHOD(ValidTRSAcceptsProperTRS) {
            const MatSIMD<float, 4, 4> trs =
                MatSIMD<float, 4, 4>::TRS(VecSIMD<float, 3>{ 1.0f, 2.0f, 3.0f }, QuaternionSIMDf::FromAxisAngle(VecSIMD<float, 3>{ 0.0f, 1.0f, 0.0f }, kPi / 4.0f),
                    VecSIMD<float, 3>{ 2.0f, 2.0f, 2.0f });

            Assert::IsTrue(ValidTRS(trs));
        }

        TEST_METHOD(ValidTRSRejectsZeroMatrix) {
            Assert::IsFalse(ValidTRS(MatSIMD<float, 4, 4>::Zero()));
        }

        TEST_METHOD(ValidTRSRejectsNonAffineBottomRow) {
            MatSIMD<float, 4, 4> mat = MatSIMD<float, 4, 4>::Identity();
            mat[3, 0] = 1.0f;

            Assert::IsFalse(ValidTRS(mat));
        }

        // =========================================================================
        // Layout
        // =========================================================================

        TEST_METHOD(RowMajorMemoryLayout) {
            const MatSIMDRowMajor<float, 2, 3> mat{ 1, 2, 3, 4, 5, 6 };

            Assert::AreEqual(1.0f, mat.Data()[0]);
            Assert::AreEqual(2.0f, mat.Data()[1]);
            Assert::AreEqual(3.0f, mat.Data()[2]);
            Assert::AreEqual(4.0f, mat.Data()[3]);
            Assert::AreEqual(5.0f, mat.Data()[4]);
            Assert::AreEqual(6.0f, mat.Data()[5]);
        }

        TEST_METHOD(ColumnMajorMemoryLayout) {
            const MatSIMDColumnMajor<float, 2, 3> mat{ 1, 2, 3, 4, 5, 6 };

            Assert::AreEqual(1.0f, mat.Data()[0]);
            Assert::AreEqual(4.0f, mat.Data()[1]);
            Assert::AreEqual(2.0f, mat.Data()[2]);
            Assert::AreEqual(5.0f, mat.Data()[3]);
            Assert::AreEqual(3.0f, mat.Data()[4]);
            Assert::AreEqual(6.0f, mat.Data()[5]);
        }

        TEST_METHOD(LayoutDoesNotChangeMathematicalIndexing) {
            const MatSIMDRowMajor<float, 2, 3> row{ 1, 2, 3, 4, 5, 6 };

            const MatSIMDColumnMajor<float, 2, 3> column{ 1, 2, 3, 4, 5, 6 };

            for (std::size_t r = 0; r < 2; ++r) {
                for (std::size_t c = 0; c < 3; ++c) {
                    Assert::IsTrue((row[r, c]) == (column[r, c]));
                }
            }
        }

        // =========================================================================
        // Layout conversion
        // =========================================================================

        TEST_METHOD(RowToColumnMajorConversion) {
            const MatSIMDRowMajor<float, 2, 3> row{ 1, 2, 3, 4, 5, 6 };

            const auto column = MatCastLayout<float, 2, 3, RowMajor, ColumnMajor>(row);

            Assert::AreEqual(1.0f, (column[0, 0]));
            Assert::AreEqual(2.0f, (column[0, 1]));
            Assert::AreEqual(3.0f, (column[0, 2]));

            Assert::AreEqual(4.0f, (column[1, 0]));
            Assert::AreEqual(5.0f, (column[1, 1]));
            Assert::AreEqual(6.0f, (column[1, 2]));

            Assert::AreEqual(1.0f, column.Data()[0]);
            Assert::AreEqual(4.0f, column.Data()[1]);
            Assert::AreEqual(2.0f, column.Data()[2]);
            Assert::AreEqual(5.0f, column.Data()[3]);
            Assert::AreEqual(3.0f, column.Data()[4]);
            Assert::AreEqual(6.0f, column.Data()[5]);
        }

        TEST_METHOD(ColumnToRowMajorConversion) {
            const MatSIMDColumnMajor<float, 2, 3> column{ 1, 2, 3, 4, 5, 6 };

            const auto row = MatCastLayout<float, 2, 3, ColumnMajor, RowMajor>(column);

            Assert::AreEqual(1.0f, (row[0, 0]));
            Assert::AreEqual(2.0f, (row[0, 1]));
            Assert::AreEqual(3.0f, (row[0, 2]));

            Assert::AreEqual(4.0f, (row[1, 0]));
            Assert::AreEqual(5.0f, (row[1, 1]));
            Assert::AreEqual(6.0f, (row[1, 2]));

            Assert::AreEqual(1.0f, row.Data()[0]);
            Assert::AreEqual(2.0f, row.Data()[1]);
            Assert::AreEqual(3.0f, row.Data()[2]);
            Assert::AreEqual(4.0f, row.Data()[3]);
            Assert::AreEqual(5.0f, row.Data()[4]);
            Assert::AreEqual(6.0f, row.Data()[5]);
        }

        TEST_METHOD(LayoutConversionConstructor) {
            const MatSIMDRowMajor<float, 2, 2> row{ 1, 2, 3, 4 };

            const MatSIMDColumnMajor<float, 2, 2> column{ row };

            Assert::AreEqual(1.0f, (column[0, 0]));
            Assert::AreEqual(2.0f, (column[0, 1]));
            Assert::AreEqual(3.0f, (column[1, 0]));
            Assert::AreEqual(4.0f, (column[1, 1]));

            Assert::AreEqual(1.0f, column.Data()[0]);
            Assert::AreEqual(3.0f, column.Data()[1]);
            Assert::AreEqual(2.0f, column.Data()[2]);
            Assert::AreEqual(4.0f, column.Data()[3]);
        }

        // =========================================================================
        // Layout-aware operations
        // =========================================================================

        TEST_METHOD(RowMajorMultiplication) {
            const MatSIMDRowMajor<float, 2, 2> a{ 1, 2, 3, 4 };

            const MatSIMDRowMajor<float, 2, 2> b{ 5, 6, 7, 8 };

            const auto result = a * b;

            Assert::AreEqual(19.0f, (result[0, 0]));
            Assert::AreEqual(22.0f, (result[0, 1]));
            Assert::AreEqual(43.0f, (result[1, 0]));
            Assert::AreEqual(50.0f, (result[1, 1]));
        }

        TEST_METHOD(ColumnMajorMultiplication) {
            const MatSIMDColumnMajor<float, 2, 2> a{ 1, 2, 3, 4 };

            const MatSIMDColumnMajor<float, 2, 2> b{ 5, 6, 7, 8 };

            const auto result = a * b;

            Assert::AreEqual(19.0f, (result[0, 0]));
            Assert::AreEqual(22.0f, (result[0, 1]));
            Assert::AreEqual(43.0f, (result[1, 0]));
            Assert::AreEqual(50.0f, (result[1, 1]));
        }

        // =========================================================================
        // Numeric conversion
        // =========================================================================

        TEST_METHOD(NumericConversion) {
            const MatSIMD<float, 2, 2> source{ 1, 2, 3, 4 };

            const MatSIMD<double, 2, 2> result{ source };

            ExpectNear(1.0, (result[0, 0]));
            ExpectNear(2.0, (result[0, 1]));
            ExpectNear(3.0, (result[1, 0]));
            ExpectNear(4.0, (result[1, 1]));
        }

        TEST_METHOD(NumericAndLayoutConversion) {
            const MatSIMDRowMajor<float, 2, 2> source{ 1, 2, 3, 4 };

            const MatSIMDColumnMajor<double, 2, 2> result{ source };

            ExpectNear(1.0, (result[0, 0]));
            ExpectNear(2.0, (result[0, 1]));
            ExpectNear(3.0, (result[1, 0]));
            ExpectNear(4.0, (result[1, 1]));

            ExpectNear(1.0, result.Data()[0]);
            ExpectNear(3.0, result.Data()[1]);
            ExpectNear(2.0, result.Data()[2]);
            ExpectNear(4.0, result.Data()[3]);
        }

        // =========================================================================
        // Formatting
        // =========================================================================

        TEST_METHOD(Formatting) {
            const MatSIMD<float, 2, 2> mat{ 1, 2, 3, 4 };

            Assert::AreEqual(std::string("[[1, 2], [3, 4]]"), std::format("{}", mat));
        }

        TEST_METHOD(FormattingFloatingPoint) {
            const MatSIMD<float, 2, 2> mat{ 1.0f, 2.5f, 3.25f, 4.0f };

            Assert::AreEqual(std::string("[[1, 2.5], [3.25, 4]]"), std::format("{}", mat));
        }
    };
}
