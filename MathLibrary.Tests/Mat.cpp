#include "pch.h"
#include "CppUnitTest.h"

#include "Headers/Mat.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace {
    constexpr float kPi = std::numbers::pi_v<float>;

    // Compares two floating-point values through Assert::AreEqual's tolerance overload.
    void ExpectNear(double expected, double actual, double tolerance = 1e-4) {
        Assert::AreEqual(expected, actual, tolerance);
    }

    template <std::floating_point T>
    void ExpectNear(const Vec3<T>& a, const Vec3<T>& b, T tolerance = static_cast<T>(1e-4)) {
        ExpectNear(a[0], b[0], tolerance);
        ExpectNear(a[1], b[1], tolerance);
        ExpectNear(a[2], b[2], tolerance);
    }
} // namespace

namespace MathLibraryTests
{
    TEST_CLASS(MatTests)
    {
    public:
        // =========================================================================
        // Construction
        // =========================================================================

        TEST_METHOD(DefaultConstruction) {
            constexpr Mat<float, 2, 3> mat;

            for (std::size_t row = 0; row < 2; ++row) {
                for (std::size_t col = 0; col < 3; ++col) {
                    Assert::AreEqual(0.0f, (mat[row, col]));
                }
            }
        }

        TEST_METHOD(ValueConstruction) {
            constexpr Mat<float, 2, 3> mat{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };

            Assert::AreEqual(1.0f, (mat[0, 0]));
            Assert::AreEqual(2.0f, (mat[0, 1]));
            Assert::AreEqual(3.0f, (mat[0, 2]));

            Assert::AreEqual(4.0f, (mat[1, 0]));
            Assert::AreEqual(5.0f, (mat[1, 1]));
            Assert::AreEqual(6.0f, (mat[1, 2]));
        }

        TEST_METHOD(ValueConstructionWithConversion) {
            constexpr Mat<float, 2, 2> mat{ 1.0f, 2.0f, 3.0f, 4.0f };

            Assert::AreEqual(1.0f, (mat[0, 0]));
            Assert::AreEqual(2.0f, (mat[0, 1]));
            Assert::AreEqual(3.0f, (mat[1, 0]));
            Assert::AreEqual(4.0f, (mat[1, 1]));
        }

        // =========================================================================
        // Access
        // =========================================================================

        TEST_METHOD(BracketAccess) {
            Mat<float, 3, 3> mat{};

            mat[1, 2] = 42.0f;

            Assert::AreEqual(42.0f, (mat[1, 2]));
        }

        TEST_METHOD(ParenthesisAccess) {
            Mat<float, 3, 3> mat{};

            mat(2, 1) = 42.0f;

            Assert::AreEqual(42.0f, (mat(2, 1)));
        }

        TEST_METHOD(ConstAccess) {
            constexpr Mat<float, 2, 2> mat{ 1.0f, 2.0f, 3.0f, 4.0f };

            Assert::AreEqual(1.0f, (mat[0, 0]));
            Assert::AreEqual(4.0f, (mat[1, 1]));
        }

        TEST_METHOD(Data) {
            Mat<float, 2, 3> mat{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };

            Assert::IsNotNull(mat.Data());

            mat.Data()[0] = 42.0f;

            Assert::AreEqual(42.0f, (mat[0, 0]));
        }

        // =========================================================================
        // Dimensions
        // =========================================================================

        TEST_METHOD(Dimensions) {
            using Matrix = Mat<float, 3, 4>;

            Assert::AreEqual(static_cast<std::size_t>(3), Matrix::Rows());
            Assert::AreEqual(static_cast<std::size_t>(4), Matrix::Cols());
            Assert::AreEqual(static_cast<std::size_t>(12), Matrix::Size());
        }

        TEST_METHOD(DimensionsAreCompileTimeConstants) {
            using Matrix = Mat<float, 3, 4>;

            static_assert(Matrix::Rows() == 3);
            static_assert(Matrix::Cols() == 4);
            static_assert(Matrix::Size() == 12);
        }

        // =========================================================================
        // MDSpan
        // =========================================================================

        TEST_METHOD(MDSpan) {
            Mat<float, 2, 3> mat{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };

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
            constexpr Mat<float, 2, 2> mat{ 1.0f, 2.0f, 3.0f, 4.0f };

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
            constexpr Mat<float, 2, 2> a{ 1, 2, 3, 4 };

            constexpr Mat<float, 2, 2> b{ 1, 2, 3, 4 };

            constexpr Mat<float, 2, 2> c{ 1, 2, 3, 5 };

            Assert::IsTrue(a == b);
            Assert::IsFalse(a == c);
        }

        TEST_METHOD(EqualityDifferentNumericTypes) {
            constexpr Mat<float, 2, 2> a{ 1, 2, 3, 4 };

            constexpr Mat<double, 2, 2> b{ 1.0, 2.0, 3.0, 4.0 };

            Assert::IsTrue(a == b);
        }

        TEST_METHOD(EqualityDifferentLayouts) {
            constexpr MatRowMajor<float, 2, 2> row{ 1, 2, 3, 4 };

            constexpr MatColumnMajor<float, 2, 2> column{ 1, 2, 3, 4 };

            Assert::IsTrue(row == column);
        }

        // =========================================================================
        // Addition
        // =========================================================================

        TEST_METHOD(Addition) {
            constexpr Mat<float, 2, 2> a{ 1, 2, 3, 4 };

            constexpr Mat<float, 2, 2> b{ 5, 6, 7, 8 };

            constexpr auto result = a + b;

            static_assert(std::is_same_v<decltype(result), const Mat<float, 2, 2>>);

            Assert::AreEqual(6.0f, (result[0, 0]));
            Assert::AreEqual(8.0f, (result[0, 1]));
            Assert::AreEqual(10.0f, (result[1, 0]));
            Assert::AreEqual(12.0f, (result[1, 1]));
        }

        TEST_METHOD(AdditionDifferentNumericTypes) {
            constexpr Mat<float, 2, 2> a{ 1, 2, 3, 4 };

            constexpr Mat<double, 2, 2> b{ 0.5, 1.5, 2.5, 3.5 };

            constexpr auto result = a + b;

            static_assert(std::is_same_v<decltype(result), const Mat<double, 2, 2>>);

            ExpectNear(1.5, (result[0, 0]));
            ExpectNear(3.5, (result[0, 1]));
            ExpectNear(5.5, (result[1, 0]));
            ExpectNear(7.5, (result[1, 1]));
        }

        TEST_METHOD(AdditionAssignment) {
            Mat<float, 2, 2> a{ 1, 2, 3, 4 };

            constexpr Mat<float, 2, 2> b{ 5, 6, 7, 8 };

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
            constexpr Mat<float, 2, 2> a{ 5, 6, 7, 8 };

            constexpr Mat<float, 2, 2> b{ 1, 2, 3, 4 };

            constexpr auto result = a - b;

            Assert::AreEqual(4.0f, (result[0, 0]));
            Assert::AreEqual(4.0f, (result[0, 1]));
            Assert::AreEqual(4.0f, (result[1, 0]));
            Assert::AreEqual(4.0f, (result[1, 1]));
        }

        TEST_METHOD(SubtractionAssignment) {
            Mat<float, 2, 2> a{ 5, 6, 7, 8 };

            constexpr Mat<float, 2, 2> b{ 1, 2, 3, 4 };

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
            constexpr Mat<float, 2, 2> a{ 1, 2, 3, 4 };

            constexpr auto result = +a;

            Assert::IsTrue(result == a);
        }

        TEST_METHOD(UnaryMinus) {
            constexpr Mat<float, 2, 2> a{ 1, -2, 3, -4 };

            constexpr auto result = -a;

            Assert::AreEqual(-1.0f, (result[0, 0]));
            Assert::AreEqual(2.0f, (result[0, 1]));
            Assert::AreEqual(-3.0f, (result[1, 0]));
            Assert::AreEqual(4.0f, (result[1, 1]));
        }

        // =========================================================================
        // Scalar multiplication
        // =========================================================================

        TEST_METHOD(ScalarMultiplication) {
            constexpr Mat<float, 2, 2> mat{ 1, 2, 3, 4 };

            constexpr auto result = mat * 2.0f;

            Assert::AreEqual(2.0f, (result[0, 0]));
            Assert::AreEqual(4.0f, (result[0, 1]));
            Assert::AreEqual(6.0f, (result[1, 0]));
            Assert::AreEqual(8.0f, (result[1, 1]));
        }

        TEST_METHOD(ScalarMultiplicationReversed) {
            constexpr Mat<float, 2, 2> mat{ 1, 2, 3, 4 };

            constexpr auto result = 2.0f * mat;

            Assert::AreEqual(2.0f, (result[0, 0]));
            Assert::AreEqual(4.0f, (result[0, 1]));
            Assert::AreEqual(6.0f, (result[1, 0]));
            Assert::AreEqual(8.0f, (result[1, 1]));
        }

        TEST_METHOD(ScalarMultiplicationDifferentType) {
            constexpr Mat<float, 2, 2> mat{ 1, 2, 3, 4 };

            constexpr auto result = mat * 0.5;

            static_assert(std::is_same_v<decltype(result), const Mat<double, 2, 2>>);

            ExpectNear(0.5, (result[0, 0]));
            ExpectNear(1.0, (result[0, 1]));
            ExpectNear(1.5, (result[1, 0]));
            ExpectNear(2.0, (result[1, 1]));
        }

        TEST_METHOD(ScalarMultiplicationAssignment) {
            Mat<float, 2, 2> mat{ 1, 2, 3, 4 };

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
            constexpr Mat<float, 2, 2> mat{ 2, 4, 6, 8 };

            constexpr auto result = mat / 2.0f;

            ExpectNear(1.0f, (result[0, 0]));
            ExpectNear(2.0f, (result[0, 1]));
            ExpectNear(3.0f, (result[1, 0]));
            ExpectNear(4.0f, (result[1, 1]));
        }

        TEST_METHOD(ScalarDivisionDifferentType) {
            constexpr Mat<float, 2, 2> mat{ 2, 4, 6, 8 };

            constexpr auto result = mat / 2.0;

            static_assert(std::is_same_v<decltype(result), const Mat<double, 2, 2>>);

            ExpectNear(1.0, (result[0, 0]));
            ExpectNear(2.0, (result[0, 1]));
            ExpectNear(3.0, (result[1, 0]));
            ExpectNear(4.0, (result[1, 1]));
        }

        TEST_METHOD(ScalarDivisionAssignment) {
            Mat<float, 2, 2> mat{ 2, 4, 6, 8 };

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
            constexpr Mat<float, 2, 3> a{ 1, 2, 3, 4, 5, 6 };

            constexpr Mat<float, 3, 2> b{ 7, 8, 9, 10, 11, 12 };

            constexpr auto result = a * b;

            static_assert(std::is_same_v<decltype(result), const Mat<float, 2, 2>>);

            Assert::AreEqual(58.0f, (result[0, 0]));
            Assert::AreEqual(64.0f, (result[0, 1]));
            Assert::AreEqual(139.0f, (result[1, 0]));
            Assert::AreEqual(154.0f, (result[1, 1]));
        }

        TEST_METHOD(MatrixMultiplicationDifferentNumericTypes) {
            constexpr Mat<float, 2, 3> a{ 1, 2, 3, 4, 5, 6 };

            constexpr Mat<double, 3, 2> b{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

            constexpr auto result = a * b;

            static_assert(std::is_same_v<decltype(result), const Mat<double, 2, 2>>);

            ExpectNear(22.0, (result[0, 0]));
            ExpectNear(28.0, (result[0, 1]));
            ExpectNear(49.0, (result[1, 0]));
            ExpectNear(64.0, (result[1, 1]));
        }

        TEST_METHOD(SquareMatrixMultiplication) {
            constexpr Mat<float, 2, 2> a{ 1, 2, 3, 4 };

            constexpr Mat<float, 2, 2> b{ 5, 6, 7, 8 };

            constexpr auto result = a * b;

            Assert::AreEqual(19.0f, (result[0, 0]));
            Assert::AreEqual(22.0f, (result[0, 1]));
            Assert::AreEqual(43.0f, (result[1, 0]));
            Assert::AreEqual(50.0f, (result[1, 1]));
        }

        // =========================================================================
        // Transpose
        // =========================================================================

        TEST_METHOD(Transpose) {
            constexpr Mat<float, 2, 3> mat{ 1, 2, 3, 4, 5, 6 };

            constexpr auto result = mat.Transpose();

            static_assert(std::is_same_v<decltype(result), const Mat<float, 3, 2>>);

            Assert::AreEqual(1.0f, (result[0, 0]));
            Assert::AreEqual(4.0f, (result[0, 1]));

            Assert::AreEqual(2.0f, (result[1, 0]));
            Assert::AreEqual(5.0f, (result[1, 1]));

            Assert::AreEqual(3.0f, (result[2, 0]));
            Assert::AreEqual(6.0f, (result[2, 1]));
        }

        TEST_METHOD(SquareTranspose) {
            constexpr Mat<float, 3, 3> mat{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            constexpr auto result = mat.Transpose();

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
            constexpr auto identity = SquareMatrix<float, 2>::Identity();

            Assert::AreEqual(1.0f, (identity[0, 0]));
            Assert::AreEqual(0.0f, (identity[0, 1]));
            Assert::AreEqual(0.0f, (identity[1, 0]));
            Assert::AreEqual(1.0f, (identity[1, 1]));
        }

        TEST_METHOD(Identity4x4) {
            constexpr auto identity = SquareMatrix<float, 4>::Identity();

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
            constexpr Mat<float, 3, 3> mat{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            constexpr auto identity = SquareMatrix<float, 3>::Identity();

            Assert::IsTrue(mat * identity == mat);
            Assert::IsTrue(identity * mat == mat);
        }

        // =========================================================================
        // Determinant
        // =========================================================================

        TEST_METHOD(Determinant1x1) {
            constexpr Mat<float, 1, 1> mat{ 5.0f };

            ExpectNear(5.0f, mat.Determinant());
        }

        TEST_METHOD(Determinant2x2) {
            constexpr Mat<float, 2, 2> mat{ 1, 2, 3, 4 };

            ExpectNear(-2.0f, mat.Determinant());
        }

        TEST_METHOD(Determinant3x3) {
            constexpr Mat<float, 3, 3> mat{ 6, 1, 1, 4, -2, 5, 2, 8, 7 };

            ExpectNear(-306.0f, mat.Determinant());
        }

        TEST_METHOD(DeterminantIdentity) {
            constexpr auto identity = SquareMatrix<float, 4>::Identity();

            ExpectNear(1.0f, identity.Determinant());
        }

        TEST_METHOD(DeterminantSingular) {
            constexpr Mat<float, 2, 2> mat{ 1, 2, 2, 4 };

            ExpectNear(0.0f, mat.Determinant());
        }

        // =========================================================================
        // Inverse
        // =========================================================================

        TEST_METHOD(Inverse2x2) {
            constexpr Mat<float, 2, 2> mat{ 4.0, 7.0, 2.0, 6.0 };

            constexpr auto inverse = mat.Inverse();

            ExpectNear(0.6, (inverse[0, 0]), 1e-5);
            ExpectNear(-0.7, (inverse[0, 1]), 1e-5);
            ExpectNear(-0.2, (inverse[1, 0]), 1e-5);
            ExpectNear(0.4, (inverse[1, 1]), 1e-5);
        }

        TEST_METHOD(InverseIdentity) {
            constexpr auto identity = SquareMatrix<float, 4>::Identity();

            constexpr auto inverse = identity.Inverse();

            Assert::IsTrue(inverse == identity);
        }

        TEST_METHOD(MatrixTimesInverseIsIdentity) {
            constexpr Mat<float, 2, 2> mat{ 4.0, 7.0, 2.0, 6.0 };

            constexpr auto inverse = mat.Inverse();
            constexpr auto result = mat * inverse;

            constexpr auto identity = SquareMatrix<float, 2>::Identity();

            for (std::size_t row = 0; row < 2; ++row) {
                for (std::size_t col = 0; col < 2; ++col) {
                    ExpectNear((identity[row, col]), (result[row, col]), 1e-5);
                }
            }
        }

        TEST_METHOD(SingularMatrixInverse) {
            constexpr Mat<float, 2, 2> mat{ 1.0, 2.0, 2.0, 4.0 };

            constexpr auto inverse = mat.Inverse();
            constexpr auto b = Mat<float, 2, 2>{};

            Assert::IsTrue(inverse == b);
        }

        // =========================================================================
        // Zero
        // =========================================================================

        TEST_METHOD(Zero) {
            constexpr auto mat = Mat4f::Zero();

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
            const Mat4f rot = Mat4f::RotationX(kPi / 2.0f);
            constexpr Vec3f v{ 0.0f, 1.0f, 0.0f };

            const Vec3f result = rot.MultiplyVector(v);

            ExpectNear(result, Vec3f{ 0.0f, 0.0f, 1.0f });
        }

        TEST_METHOD(RotationYRotatesZIntoX) {
            const Mat4f rot = Mat4f::RotationY(kPi / 2.0f);
            constexpr Vec3f v{ 0.0f, 0.0f, 1.0f };

            const Vec3f result = rot.MultiplyVector(v);

            ExpectNear(result, Vec3f{ 1.0f, 0.0f, 0.0f });
        }

        TEST_METHOD(RotationZRotatesXIntoY) {
            const Mat4f rot = Mat4f::RotationZ(kPi / 2.0f);
            constexpr Vec3f v{ 1.0f, 0.0f, 0.0f };

            const Vec3f result = rot.MultiplyVector(v);

            ExpectNear(result, Vec3f{ 0.0f, 1.0f, 0.0f });
        }

        TEST_METHOD(RotationXByZeroIsIdentity) {
            const Mat4f rot = Mat4f::RotationX(0.0f);
            const Mat4f identity = Mat4f::Identity();

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
            const Mat4f mat = Mat4f::Scale(Vec3f{ 2.0f, 3.0f, 4.0f });
            constexpr Vec3f point{ 1.0f, 1.0f, 1.0f };

            const Vec3f result = mat.MultiplyPoint(point);

            ExpectNear(result, Vec3f{ 2.0f, 3.0f, 4.0f });
        }

        TEST_METHOD(TranslateFactoryTranslatesPoint) {
            const Mat4f mat = Mat4f::Translate(Vec3f{ 1.0f, 2.0f, 3.0f });
            constexpr Vec3f point{ 0.0f, 0.0f, 0.0f };

            const Vec3f result = mat.MultiplyPoint(point);

            ExpectNear(result, Vec3f{ 1.0f, 2.0f, 3.0f });
        }

        TEST_METHOD(TranslateFactoryDoesNotAffectVector) {
            const Mat4f mat = Mat4f::Translate(Vec3f{ 1.0f, 2.0f, 3.0f });
            constexpr Vec3f direction{ 5.0f, 0.0f, 0.0f };

            const Vec3f result = mat.MultiplyVector(direction);

            ExpectNear(result, direction);
        }

        // =========================================================================
        // Rotate (from quaternion)
        // =========================================================================

        TEST_METHOD(RotateMatchesQuaternionRotateVector) {
            const Quaternionf rotation = Quaternionf::FromAxisAngle(Vec3f{ 0.0f, 1.0f, 0.0f }, kPi / 3.0f);
            const Mat4f mat = Mat4f::Rotate(rotation);

            constexpr Vec3f v{ 1.0f, 0.0f, 0.0f };

            ExpectNear(mat.MultiplyVector(v), rotation.RotateVector(v));
        }

        TEST_METHOD(RotateIdentityQuaternionIsIdentityMatrix) {
            const Mat4f mat = Mat4f::Rotate(Quaternionf::Identity());
            const Mat4f identity = Mat4f::Identity();

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
            constexpr Vec3f translation{ 10.0f, 0.0f, 0.0f };
            const Quaternionf rotation = Quaternionf::Identity();
            constexpr Vec3f scale{ 2.0f, 1.0f, 1.0f };

            const Mat4f trs = Mat4f::TRS(translation, rotation, scale);
            constexpr Vec3f localPoint{ 1.0f, 0.0f, 0.0f };

            const Vec3f worldPoint = trs.MultiplyPoint(localPoint);

            ExpectNear(worldPoint, Vec3f{ 12.0f, 0.0f, 0.0f });
        }

        TEST_METHOD(TRSWithIdentityComponentsIsIdentity) {
            const Mat4f trs = Mat4f::TRS(Vec3f{ 0.0f, 0.0f, 0.0f }, Quaternionf::Identity(), Vec3f{ 1.0f, 1.0f, 1.0f });
            const Mat4f identity = Mat4f::Identity();

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
            const Mat4f persp = Mat4f::Perspective(kPi / 2.0f, 1.0f, 0.1f, 100.0f);

            const Vec4f nearPoint{ 0.0f, 0.0f, -0.1f, 1.0f };
            const Vec4f clip = persp * nearPoint;

            ExpectNear(-1.0f, clip[2] / clip[3], 1e-4f);
        }

        TEST_METHOD(PerspectiveMapsFarPlaneToPositiveOneNDC) {
            const Mat4f persp = Mat4f::Perspective(kPi / 2.0f, 1.0f, 0.1f, 100.0f);

            const Vec4f farPoint{ 0.0f, 0.0f, -100.0f, 1.0f };
            const Vec4f clip = persp * farPoint;

            ExpectNear(1.0f, clip[2] / clip[3], 1e-4f);
        }

        TEST_METHOD(OrthoMapsNearPlaneToNegativeOne) {
            const Mat4f ortho = Mat4f::Ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);

            const Vec3f nearPoint{ 0.0f, 0.0f, -0.1f };
            const Vec3f mapped = ortho.MultiplyPoint(nearPoint);

            ExpectNear(-1.0f, mapped[2], 1e-4f);
        }

        TEST_METHOD(OrthoMapsFarPlaneToPositiveOne) {
            const Mat4f ortho = Mat4f::Ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);

            const Vec3f farPoint{ 0.0f, 0.0f, -100.0f };
            const Vec3f mapped = ortho.MultiplyPoint(farPoint);

            ExpectNear(1.0f, mapped[2], 1e-4f);
        }

        TEST_METHOD(OrthoMapsSideBoundsToUnitRange) {
            const Mat4f ortho = Mat4f::Ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);

            const Vec3f corner{ 1.0f, 1.0f, -0.1f };
            const Vec3f mapped = ortho.MultiplyPoint(corner);

            ExpectNear(1.0f, mapped[0], 1e-4f);
            ExpectNear(1.0f, mapped[1], 1e-4f);
        }

        TEST_METHOD(LookAtMapsEyeToOrigin) {
            constexpr Vec3f eye{ 0.0f, 0.0f, 5.0f };
            constexpr Vec3f target{ 0.0f, 0.0f, 0.0f };
            constexpr Vec3f up{ 0.0f, 1.0f, 0.0f };

            const Mat4f lookAt = Mat4f::LookAt(eye, target, up);
            const Vec3f eyeInView = lookAt.MultiplyPoint(eye);

            ExpectNear(eyeInView, Vec3f{ 0.0f, 0.0f, 0.0f });
        }

        TEST_METHOD(LookAtRotationIsOrthonormal) {
            constexpr Vec3f eye{ 0.0f, 0.0f, 5.0f };
            constexpr Vec3f target{ 0.0f, 0.0f, 0.0f };
            constexpr Vec3f up{ 0.0f, 1.0f, 0.0f };

            const Mat4f lookAt = Mat4f::LookAt(eye, target, up);

            const Vec3f r0{ lookAt[0, 0], lookAt[0, 1], lookAt[0, 2] };
            const Vec3f r1{ lookAt[1, 0], lookAt[1, 1], lookAt[1, 2] };
            const Vec3f r2{ lookAt[2, 0], lookAt[2, 1], lookAt[2, 2] };

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
            const Mat4f mat = Mat4f::Translate(Vec3f{ 1.0f, 2.0f, 3.0f });
            constexpr Vec3f point{ 1.0f, 1.0f, 1.0f };

            const Vec3f result = mat.MultiplyPoint(point);

            ExpectNear(result, Vec3f{ 2.0f, 3.0f, 4.0f });
        }

        TEST_METHOD(MultiplyVectorIgnoresTranslation) {
            const Mat4f mat = Mat4f::Translate(Vec3f{ 1.0f, 2.0f, 3.0f });
            constexpr Vec3f vector{ 1.0f, 1.0f, 1.0f };

            const Vec3f result = mat.MultiplyVector(vector);

            ExpectNear(result, vector);
        }

        // =========================================================================
        // Extract Position / Scale / Rotation
        // =========================================================================

        TEST_METHOD(ExtractPositionFromTRS) {
            constexpr Vec3f translation{ 1.0f, 2.0f, 3.0f };
            const Mat4f trs = Mat4f::TRS(translation, Quaternionf::Identity(), Vec3f{ 1.0f, 1.0f, 1.0f });

            ExpectNear(trs.ExtractPosition(), translation);
        }

        TEST_METHOD(ExtractScaleFromTRS) {
            constexpr Vec3f scale{ 2.0f, 3.0f, 4.0f };
            const Mat4f trs = Mat4f::TRS(Vec3f{ 0.0f, 0.0f, 0.0f }, Quaternionf::Identity(), scale);

            ExpectNear(trs.ExtractScale(), scale);
        }

        TEST_METHOD(ExtractRotationFromTRS) {
            const Quaternionf rotation = Quaternionf::FromAxisAngle(Vec3f{ 0.0f, 1.0f, 0.0f }, kPi / 4.0f);
            const Mat4f trs = Mat4f::TRS(Vec3f{ 0.0f, 0.0f, 0.0f }, rotation, Vec3f{ 1.0f, 1.0f, 1.0f });

            const Quaternionf extracted = trs.ExtractRotation();

            ExpectNear(rotation.X(), extracted.X(), 1e-4f);
            ExpectNear(rotation.Y(), extracted.Y(), 1e-4f);
            ExpectNear(rotation.Z(), extracted.Z(), 1e-4f);
            ExpectNear(rotation.W(), extracted.W(), 1e-4f);
        }

        TEST_METHOD(ExtractRoundTripsFullTRS) {
            constexpr Vec3f translation{ 5.0f, -2.0f, 1.0f };
            const Quaternionf rotation = Quaternionf::FromAxisAngle(Vec3f{ 1.0f, 0.0f, 0.0f }, kPi / 6.0f);
            constexpr Vec3f scale{ 2.0f, 1.0f, 3.0f };

            const Mat4f trs = Mat4f::TRS(translation, rotation, scale);

            ExpectNear(trs.ExtractPosition(), translation);
            ExpectNear(trs.ExtractScale(), scale);
        }

        // =========================================================================
        // Matrix-vector multiplication
        // =========================================================================

        TEST_METHOD(IdentityTimesVectorIsUnchanged) {
            const Mat4f identity = Mat4f::Identity();
            constexpr Vec4f v{ 1.0f, 2.0f, 3.0f, 1.0f };

            const Vec4f result = identity * v;

            ExpectNear(1.0f, result[0]);
            ExpectNear(2.0f, result[1]);
            ExpectNear(3.0f, result[2]);
            ExpectNear(1.0f, result[3]);
        }

        TEST_METHOD(MatrixTimesVectorAppliesTransform) {
            const Mat4f translate = Mat4f::Translate(Vec3f{ 1.0f, 2.0f, 3.0f });
            constexpr Vec4f v{ 0.0f, 0.0f, 0.0f, 1.0f };

            const Vec4f result = translate * v;

            ExpectNear(1.0f, result[0]);
            ExpectNear(2.0f, result[1]);
            ExpectNear(3.0f, result[2]);
            ExpectNear(1.0f, result[3]);
        }

        // =========================================================================
        // ToRadians
        // =========================================================================

        TEST_METHOD(ToRadiansZero) {
            ExpectNear(0.0f, ToRadians(0.0f));
        }

        TEST_METHOD(ToRadiansOneEighty) {
            ExpectNear(kPi, ToRadians(180.0f));
        }

        TEST_METHOD(ToRadiansNinety) {
            ExpectNear(kPi / 2.0f, ToRadians(90.0f), 1e-5f);
        }

        // =========================================================================
        // ValidTRS
        // =========================================================================

        TEST_METHOD(ValidTRSAcceptsIdentity) {
            Assert::IsTrue(ValidTRS(Mat4f::Identity()));
        }

        TEST_METHOD(ValidTRSAcceptsProperTRS) {
            const Mat4f trs =
                Mat4f::TRS(Vec3f{ 1.0f, 2.0f, 3.0f }, Quaternionf::FromAxisAngle(Vec3f{ 0.0f, 1.0f, 0.0f }, kPi / 4.0f),
                    Vec3f{ 2.0f, 2.0f, 2.0f });

            Assert::IsTrue(ValidTRS(trs));
        }

        TEST_METHOD(ValidTRSRejectsZeroMatrix) {
            Assert::IsFalse(ValidTRS(Mat4f::Zero()));
        }

        TEST_METHOD(ValidTRSRejectsNonAffineBottomRow) {
            Mat4f mat = Mat4f::Identity();
            mat[3, 0] = 1.0f;

            Assert::IsFalse(ValidTRS(mat));
        }

        // =========================================================================
        // Layout
        // =========================================================================

        TEST_METHOD(RowMajorMemoryLayout) {
            constexpr MatRowMajor<float, 2, 3> mat{ 1, 2, 3, 4, 5, 6 };

            Assert::AreEqual(1.0f, mat.Data()[0]);
            Assert::AreEqual(2.0f, mat.Data()[1]);
            Assert::AreEqual(3.0f, mat.Data()[2]);
            Assert::AreEqual(4.0f, mat.Data()[3]);
            Assert::AreEqual(5.0f, mat.Data()[4]);
            Assert::AreEqual(6.0f, mat.Data()[5]);
        }

        TEST_METHOD(ColumnMajorMemoryLayout) {
            constexpr MatColumnMajor<float, 2, 3> mat{ 1, 2, 3, 4, 5, 6 };

            Assert::AreEqual(1.0f, mat.Data()[0]);
            Assert::AreEqual(4.0f, mat.Data()[1]);
            Assert::AreEqual(2.0f, mat.Data()[2]);
            Assert::AreEqual(5.0f, mat.Data()[3]);
            Assert::AreEqual(3.0f, mat.Data()[4]);
            Assert::AreEqual(6.0f, mat.Data()[5]);
        }

        TEST_METHOD(LayoutDoesNotChangeMathematicalIndexing) {
            constexpr MatRowMajor<float, 2, 3> row{ 1, 2, 3, 4, 5, 6 };

            constexpr MatColumnMajor<float, 2, 3> column{ 1, 2, 3, 4, 5, 6 };

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
            constexpr MatRowMajor<float, 2, 3> row{ 1, 2, 3, 4, 5, 6 };

            constexpr auto column = MatCastLayout<float, 2, 3, RowMajor, ColumnMajor>(row);

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
            constexpr MatColumnMajor<float, 2, 3> column{ 1, 2, 3, 4, 5, 6 };

            constexpr auto row = MatCastLayout<float, 2, 3, ColumnMajor, RowMajor>(column);

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
            constexpr MatRowMajor<float, 2, 2> row{ 1, 2, 3, 4 };

            constexpr MatColumnMajor<float, 2, 2> column{ row };

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
            constexpr MatRowMajor<float, 2, 2> a{ 1, 2, 3, 4 };

            constexpr MatRowMajor<float, 2, 2> b{ 5, 6, 7, 8 };

            constexpr auto result = a * b;

            Assert::AreEqual(19.0f, (result[0, 0]));
            Assert::AreEqual(22.0f, (result[0, 1]));
            Assert::AreEqual(43.0f, (result[1, 0]));
            Assert::AreEqual(50.0f, (result[1, 1]));
        }

        TEST_METHOD(ColumnMajorMultiplication) {
            constexpr MatColumnMajor<float, 2, 2> a{ 1, 2, 3, 4 };

            constexpr MatColumnMajor<float, 2, 2> b{ 5, 6, 7, 8 };

            constexpr auto result = a * b;

            Assert::AreEqual(19.0f, (result[0, 0]));
            Assert::AreEqual(22.0f, (result[0, 1]));
            Assert::AreEqual(43.0f, (result[1, 0]));
            Assert::AreEqual(50.0f, (result[1, 1]));
        }

        // =========================================================================
        // Numeric conversion
        // =========================================================================

        TEST_METHOD(NumericConversion) {
            constexpr Mat<float, 2, 2> source{ 1, 2, 3, 4 };

            constexpr Mat<double, 2, 2> result{ source };

            ExpectNear(1.0, (result[0, 0]));
            ExpectNear(2.0, (result[0, 1]));
            ExpectNear(3.0, (result[1, 0]));
            ExpectNear(4.0, (result[1, 1]));
        }

        TEST_METHOD(NumericAndLayoutConversion) {
            constexpr MatRowMajor<float, 2, 2> source{ 1, 2, 3, 4 };

            constexpr MatColumnMajor<double, 2, 2> result{ source };

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
        // Aliases
        // =========================================================================

        TEST_METHOD(Aliases) {
            static_assert(std::is_same_v<Mat2f, SquareMatrix<float, 2>>);
            static_assert(std::is_same_v<Mat3f, SquareMatrix<float, 3>>);
            static_assert(std::is_same_v<Mat4f, SquareMatrix<float, 4>>);

            static_assert(std::is_same_v<SquareMatrixRowMajor<float, 4>, Mat<float, 4, 4, RowMajor>>);
            static_assert(std::is_same_v<SquareMatrixColumnMajor<float, 4>, Mat<float, 4, 4, ColumnMajor>>);

            static_assert(std::is_same_v<MatRowMajor<float, 3, 4>, Mat<float, 3, 4, RowMajor>>);
            static_assert(std::is_same_v<MatColumnMajor<float, 3, 4>, Mat<float, 3, 4, ColumnMajor>>);
        }

        // =========================================================================
        // Formatting
        // =========================================================================

        TEST_METHOD(Formatting) {
            constexpr Mat<float, 2, 2> mat{ 1, 2, 3, 4 };

            Assert::AreEqual(std::string("[[1, 2], [3, 4]]"), std::format("{}", mat));
        }

        TEST_METHOD(FormattingFloatingPoint) {
            constexpr Mat<float, 2, 2> mat{ 1.0f, 2.5f, 3.25f, 4.0f };

            Assert::AreEqual(std::string("[[1, 2.5], [3.25, 4]]"), std::format("{}", mat));
        }
    };
}
