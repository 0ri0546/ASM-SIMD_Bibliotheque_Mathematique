// SPDX-License-Identifier: MIT

#include "pch.h"
#include "CppUnitTest.h"

#include "VecSIMD.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace {
    // Compares two floating-point values through Assert::AreEqual's tolerance overload.
    void ExpectNear(double expected, double actual, double tolerance = 1e-4) {
        Assert::AreEqual(expected, actual, tolerance);
    }
} // namespace

namespace MathLibraryTests
{
    TEST_CLASS(VecSIMDTests)
    {
    public:
        // =========================================================================
        // Construction
        // =========================================================================

        TEST_METHOD(DefaultConstruction) {
            constexpr VecSIMD<float, 3> vec{};

            Assert::AreEqual(0.0f, vec[0]);
            Assert::AreEqual(0.0f, vec[1]);
            Assert::AreEqual(0.0f, vec[2]);
        }

        TEST_METHOD(ValueConstruction) {
            constexpr VecSIMD<float, 3> vec{ 1, 2, 3 };

            Assert::AreEqual(1.0f, vec[0]);
            Assert::AreEqual(2.0f, vec[1]);
            Assert::AreEqual(3.0f, vec[2]);
        }

        TEST_METHOD(DifferentDimensions) {
            constexpr VecSIMD<float, 2> vec2{ 1, 2 };
            constexpr VecSIMD<float, 3> vec3{ 1, 2, 3 };
            constexpr VecSIMD<float, 4> vec4{ 1, 2, 3, 4 };

            Assert::AreEqual(static_cast<std::size_t>(2), static_cast<std::size_t>(vec2.Size()));
            Assert::AreEqual(static_cast<std::size_t>(3), static_cast<std::size_t>(vec3.Size()));
            Assert::AreEqual(static_cast<std::size_t>(4), static_cast<std::size_t>(vec4.Size()));

            Assert::AreEqual(1.0f, vec2[0]);
            Assert::AreEqual(2.0f, vec2[1]);

            Assert::AreEqual(1.0f, vec3[0]);
            Assert::AreEqual(2.0f, vec3[1]);
            Assert::AreEqual(3.0f, vec3[2]);

            Assert::AreEqual(1.0f, vec4[0]);
            Assert::AreEqual(2.0f, vec4[1]);
            Assert::AreEqual(3.0f, vec4[2]);
            Assert::AreEqual(4.0f, vec4[3]);
        }

        TEST_METHOD(MixedValueTypes) {
            constexpr VecSIMD<float, 3> vec{ 1, 2.5, 3 };

            ExpectNear(1.0f, vec[0]);
            ExpectNear(2.5f, vec[1]);
            ExpectNear(3.0f, vec[2]);
        }

        // =========================================================================
        // Access
        // =========================================================================

        TEST_METHOD(ElementAccess) {
            VecSIMD<float, 3> vec{ 1, 2, 3 };

            vec[0] = 10;
            vec[1] = 20;
            vec[2] = 30;

            Assert::AreEqual(10.0f, vec[0]);
            Assert::AreEqual(20.0f, vec[1]);
            Assert::AreEqual(30.0f, vec[2]);
        }

        TEST_METHOD(ConstElementAccess) {
            constexpr VecSIMD<float, 3> vec{ 1, 2, 3 };

            Assert::AreEqual(1.0f, vec[0]);
            Assert::AreEqual(2.0f, vec[1]);
            Assert::AreEqual(3.0f, vec[2]);
        }

        TEST_METHOD(Data) {
            VecSIMD<float, 3> vec{ 1, 2, 3 };

            float* data = vec.Data();

            Assert::IsNotNull(data);

            data[0] = 10;
            data[1] = 20;
            data[2] = 30;

            Assert::AreEqual(10.0f, vec[0]);
            Assert::AreEqual(20.0f, vec[1]);
            Assert::AreEqual(30.0f, vec[2]);
        }

        TEST_METHOD(Size) {
            Assert::AreEqual(static_cast<std::size_t>(2), static_cast<std::size_t>((VecSIMD<float, 2>::Size())));
            Assert::AreEqual(static_cast<std::size_t>(3), static_cast<std::size_t>((VecSIMD<float, 3>::Size())));
            Assert::AreEqual(static_cast<std::size_t>(4), static_cast<std::size_t>((VecSIMD<float, 4>::Size())));
            Assert::AreEqual(static_cast<std::size_t>(8), static_cast<std::size_t>((VecSIMD<float, 8>::Size())));
        }

        // =========================================================================
        // Equality
        // =========================================================================

        TEST_METHOD(Equality) {
            constexpr VecSIMD<float, 3> a{ 1, 2, 3 };
            constexpr VecSIMD<float, 3> b{ 1, 2, 3 };

            Assert::IsTrue(a == b);
        }

        TEST_METHOD(Inequality) {
            constexpr VecSIMD<float, 3> a{ 1, 2, 3 };
            constexpr VecSIMD<float, 3> b{ 1, 2, 4 };

            Assert::IsFalse(a == b);
        }

        TEST_METHOD(EqualityDifferentTypes) {
            constexpr VecSIMD<float, 3> a{ 1, 2, 3 };
            constexpr VecSIMD<double, 3> b{ 1.0, 2.0, 3.0 };

            Assert::IsTrue(a == b);
        }

        // =========================================================================
        // Addition
        // =========================================================================

        TEST_METHOD(Addition) {
            constexpr VecSIMD<float, 3> a{ 1, 2, 3 };
            constexpr VecSIMD<float, 3> b{ 4, 5, 6 };

            const auto result = a + b;

            Assert::AreEqual(5.0f, result[0]);
            Assert::AreEqual(7.0f, result[1]);
            Assert::AreEqual(9.0f, result[2]);
        }

        TEST_METHOD(AdditionCommonType) {
            constexpr VecSIMD<float, 3> a{ 1, 2, 3 };
            constexpr VecSIMD<double, 3> b{ 0.5, 1.5, 2.5 };

            const auto result = a + b;

            static_assert(std::is_same_v<decltype(result), const VecSIMD<double, 3>>);

            ExpectNear(1.5, result[0]);
            ExpectNear(3.5, result[1]);
            ExpectNear(5.5, result[2]);
        }

        TEST_METHOD(AdditionAssignment) {
            VecSIMD<float, 3> a{ 1, 2, 3 };
            constexpr VecSIMD<float, 3> b{ 4, 5, 6 };

            a += b;

            Assert::AreEqual(5.0f, a[0]);
            Assert::AreEqual(7.0f, a[1]);
            Assert::AreEqual(9.0f, a[2]);
        }

        TEST_METHOD(MixedTypeAdditionAssignment) {
            VecSIMD<double, 3> a{ 1.0, 2.0, 3.0 };
            constexpr VecSIMD<float, 3> b{ 4, 5, 6 };

            a += b;

            ExpectNear(5.0, a[0]);
            ExpectNear(7.0, a[1]);
            ExpectNear(9.0, a[2]);
        }

        // =========================================================================
        // Subtraction
        // =========================================================================

        TEST_METHOD(Subtraction) {
            constexpr VecSIMD<float, 3> a{ 5, 7, 9 };
            constexpr VecSIMD<float, 3> b{ 1, 2, 3 };

            auto result = a - b;

            Assert::AreEqual(4.0f, result[0]);
            Assert::AreEqual(5.0f, result[1]);
            Assert::AreEqual(6.0f, result[2]);
        }

        TEST_METHOD(SubtractionCommonType) {
            constexpr VecSIMD<float, 3> a{ 5, 7, 9 };
            constexpr VecSIMD<double, 3> b{ 0.5, 1.5, 2.5 };

            const auto result = a - b;

            static_assert(std::is_same_v<decltype(result), const VecSIMD<double, 3>>);

            ExpectNear(4.5, result[0]);
            ExpectNear(5.5, result[1]);
            ExpectNear(6.5, result[2]);
        }

        TEST_METHOD(SubtractionAssignment) {
            VecSIMD<float, 3> a{ 5, 7, 9 };
            constexpr VecSIMD<float, 3> b{ 1, 2, 3 };

            a -= b;

            Assert::AreEqual(4.0f, a[0]);
            Assert::AreEqual(5.0f, a[1]);
            Assert::AreEqual(6.0f, a[2]);
        }

        // =========================================================================
        // Component-wise multiplication
        // =========================================================================

        TEST_METHOD(ComponentMultiplication) {
            constexpr VecSIMD<float, 3> a{ 1, 2, 3 };
            constexpr VecSIMD<float, 3> b{ 4, 5, 6 };

            const auto result = a * b;

            Assert::AreEqual(4.0f, result[0]);
            Assert::AreEqual(10.0f, result[1]);
            Assert::AreEqual(18.0f, result[2]);
        }

        TEST_METHOD(ComponentMultiplicationCommonType) {
            constexpr VecSIMD<float, 3> a{ 2, 3, 4 };
            constexpr VecSIMD<double, 3> b{ 0.5, 1.5, 2.5 };

            const auto result = a * b;

            static_assert(std::is_same_v<decltype(result), const VecSIMD<double, 3>>);

            ExpectNear(1.0, result[0]);
            ExpectNear(4.5, result[1]);
            ExpectNear(10.0, result[2]);
        }

        TEST_METHOD(ComponentMultiplicationAssignment) {
            VecSIMD<float, 3> a{ 2, 3, 4 };
            constexpr VecSIMD<float, 3> b{ 5, 6, 7 };

            a *= b;

            Assert::AreEqual(10.0f, a[0]);
            Assert::AreEqual(18.0f, a[1]);
            Assert::AreEqual(28.0f, a[2]);
        }

        // =========================================================================
        // Component-wise division
        // =========================================================================

        TEST_METHOD(ComponentDivision) {
            constexpr VecSIMD<float, 3> a{ 10, 20, 30 };
            constexpr VecSIMD<float, 3> b{ 2, 4, 5 };

            const auto result = a / b;

            Assert::AreEqual(5.0f, result[0]);
            Assert::AreEqual(5.0f, result[1]);
            Assert::AreEqual(6.0f, result[2]);
        }

        TEST_METHOD(ComponentDivisionCommonType) {
            constexpr VecSIMD<float, 3> a{ 1, 3, 5 };
            constexpr VecSIMD<double, 3> b{ 2.0, 2.0, 2.0 };

            const auto result = a / b;

            static_assert(std::is_same_v<decltype(result), const VecSIMD<double, 3>>);

            ExpectNear(0.5, result[0]);
            ExpectNear(1.5, result[1]);
            ExpectNear(2.5, result[2]);
        }

        TEST_METHOD(ComponentDivisionAssignment) {
            VecSIMD<float, 3> a{ 10, 20, 30 };
            constexpr VecSIMD<float, 3> b{ 2, 4, 5 };

            a /= b;

            Assert::AreEqual(5.0f, a[0]);
            Assert::AreEqual(5.0f, a[1]);
            Assert::AreEqual(6.0f, a[2]);
        }

        // =========================================================================
        // Scalar multiplication
        // =========================================================================

        TEST_METHOD(ScalarMultiplication) {
            constexpr VecSIMD<float, 3> vec{ 1, 2, 3 };

            const auto result = vec * 2.0f;

            Assert::AreEqual(2.0f, result[0]);
            Assert::AreEqual(4.0f, result[1]);
            Assert::AreEqual(6.0f, result[2]);
        }

        TEST_METHOD(ScalarMultiplicationCommonType) {
            constexpr VecSIMD<float, 3> vec{ 1, 2, 3 };

            const auto result = vec * 0.5;

            static_assert(std::is_same_v<decltype(result), const VecSIMD<double, 3>>);

            ExpectNear(0.5, result[0]);
            ExpectNear(1.0, result[1]);
            ExpectNear(1.5, result[2]);
        }

        TEST_METHOD(ScalarMultiplicationReversed) {
            constexpr VecSIMD<float, 3> vec{ 1, 2, 3 };

            const auto result = 2.0f * vec;

            Assert::AreEqual(2.0f, result[0]);
            Assert::AreEqual(4.0f, result[1]);
            Assert::AreEqual(6.0f, result[2]);
        }

        TEST_METHOD(ScalarMultiplicationAssignment) {
            VecSIMD<float, 3> vec{ 1, 2, 3 };

            vec *= 3.0f;

            Assert::AreEqual(3.0f, vec[0]);
            Assert::AreEqual(6.0f, vec[1]);
            Assert::AreEqual(9.0f, vec[2]);
        }

        // =========================================================================
        // Scalar division
        // =========================================================================

        TEST_METHOD(ScalarDivision) {
            constexpr VecSIMD<float, 3> vec{ 10, 20, 30 };

            const auto result = vec / 2.0f;

            Assert::AreEqual(5.0f, result[0]);
            Assert::AreEqual(10.0f, result[1]);
            Assert::AreEqual(15.0f, result[2]);
        }

        TEST_METHOD(ScalarDivisionCommonType) {
            constexpr VecSIMD<float, 3> vec{ 1, 2, 3 };

            const auto result = vec / 2.0;

            static_assert(std::is_same_v<decltype(result), const VecSIMD<double, 3>>);

            ExpectNear(0.5, result[0]);
            ExpectNear(1.0, result[1]);
            ExpectNear(1.5, result[2]);
        }

        TEST_METHOD(ScalarDivisionAssignment) {
            VecSIMD<float, 3> vec{ 10, 20, 30 };

            vec /= 2.0f;

            Assert::AreEqual(5.0f, vec[0]);
            Assert::AreEqual(10.0f, vec[1]);
            Assert::AreEqual(15.0f, vec[2]);
        }

        // =========================================================================
        // Unary operators
        // =========================================================================

        TEST_METHOD(UnaryPlus) {
            constexpr VecSIMD<float, 3> vec{ 1, 2, 3 };

            const auto result = +vec;

            Assert::AreEqual(1.0f, result[0]);
            Assert::AreEqual(2.0f, result[1]);
            Assert::AreEqual(3.0f, result[2]);
        }

        TEST_METHOD(UnaryMinus) {
            constexpr VecSIMD<float, 3> vec{ 1, -2, 3 };

            const auto result = -vec;

            Assert::AreEqual(-1.0f, result[0]);
            Assert::AreEqual(2.0f, result[1]);
            Assert::AreEqual(-3.0f, result[2]);
        }

        // =========================================================================
        // Aliases
        // =========================================================================

        TEST_METHOD(VecSIMDAliases) {
            static_assert(std::is_same_v<VecSIMD<double, 2>, VecSIMD<double, 2>>);
            static_assert(std::is_same_v<VecSIMD<double, 3>, VecSIMD<double, 3>>);
            static_assert(std::is_same_v<VecSIMD<double, 4>, VecSIMD<double, 4>>);

            static_assert(std::is_same_v<VecSIMD<float, 2>, VecSIMD<float, 2>>);
            static_assert(std::is_same_v<VecSIMD<float, 3>, VecSIMD<float, 3>>);
            static_assert(std::is_same_v<VecSIMD<float, 4>, VecSIMD<float, 4>>);
        }

        // =========================================================================
        // Common type
        // =========================================================================

        TEST_METHOD(CommonType) {
            static_assert(std::is_same_v<VecSIMDCommon<float, double, 2>, VecSIMD<double, 2>>);

            static_assert(std::is_same_v<VecSIMDCommon<double, float, 3>, VecSIMD<double, 3>>);
        }

        // =========================================================================
        // Constexpr
        // =========================================================================

        TEST_METHOD(ConstexprConstruction) {
            constexpr VecSIMD<float, 3> vec{ 1, 2, 3 };

            static_assert(vec[0] == 1);
            static_assert(vec[1] == 2);
            static_assert(vec[2] == 3);
        }

        TEST_METHOD(ConstexprArithmetic) {
            constexpr VecSIMD<float, 3> a{ 1, 2, 3 };
            constexpr VecSIMD<float, 3> b{ 4, 5, 6 };

            const auto sum = a + b;
            const auto difference = a - b;
            const auto product = a * b;

            sum[0] == 5;
            sum[1] == 7;
            sum[2] == 9;

            difference[0] == -3;
            difference[1] == -3;
            difference[2] == -3;

            product[0] == 4;
            product[1] == 10;
            product[2] == 18;
        }

        // =========================================================================
        // Length / Normalize
        // =========================================================================

        TEST_METHOD(LengthSquared) {
            constexpr VecSIMD<float, 3> vec{ 3, 4, 0 };

            Assert::AreEqual(25.0f, vec.LengthSquared());
        }

        TEST_METHOD(Length) {
            const VecSIMD<float, 3> vec{ 3.0f, 4.0f, 0.0f };

            ExpectNear(5.0f, vec.Length());
        }

        TEST_METHOD(LengthOfZeroVecSIMDtor) {
            constexpr VecSIMD<float, 3> vec{ 0.0f, 0.0f, 0.0f };

            ExpectNear(0.0f, vec.LengthSquared());
            ExpectNear(0.0f, vec.Length());
        }

        TEST_METHOD(Normalized) {
            const VecSIMD<float, 3> vec{ 3.0f, 4.0f, 0.0f };

            const auto result = vec.Normalized();

            ExpectNear(1.0f, result.Length());
            ExpectNear(0.6f, result[0]);
            ExpectNear(0.8f, result[1]);
            ExpectNear(0.0f, result[2]);
        }

        TEST_METHOD(NormalizedDoesNotMutate) {
            const VecSIMD<float, 3> vec{ 3.0f, 4.0f, 0.0f };

            [[maybe_unused]] const auto result = vec.Normalized();

            ExpectNear(3.0f, vec[0]);
            ExpectNear(4.0f, vec[1]);
        }

        TEST_METHOD(NormalizedZeroVecSIMDtor) {
            constexpr VecSIMD<float, 3> vec{ 0.0f, 0.0f, 0.0f };

            const auto result = vec.Normalized();

            ExpectNear(0.0f, result[0]);
            ExpectNear(0.0f, result[1]);
            ExpectNear(0.0f, result[2]);
        }

        TEST_METHOD(Normalize) {
            VecSIMD<float, 3> vec{ 3.0f, 4.0f, 0.0f };

            vec.Normalize();

            ExpectNear(1.0f, vec.Length());
            ExpectNear(0.6f, vec[0]);
            ExpectNear(0.8f, vec[1]);
        }

        // =========================================================================
        // Dot
        // =========================================================================

        TEST_METHOD(Dot) {
            constexpr VecSIMD<float, 3> a{ 1, 2, 3 };
            constexpr VecSIMD<float, 3> b{ 4, 5, 6 };

            Assert::AreEqual(32.0f, ::Dot(a, b));
        }

        TEST_METHOD(DotOfOrthogonalVecSIMDtors) {
            constexpr VecSIMD<float, 3> a{ 1, 0, 0 };
            constexpr VecSIMD<float, 3> b{ 0, 1, 0 };

            Assert::AreEqual(0.0f, ::Dot(a, b));
        }

        TEST_METHOD(DotCommonType) {
            constexpr VecSIMD<float, 3> a{ 1, 2, 3 };
            constexpr VecSIMD<double, 3> b{ 0.5, 0.5, 0.5 };

            const auto result = ::Dot(a, b);

            static_assert(std::is_same_v<decltype(result), const double>);

            ExpectNear(3.0, result);
        }

        // =========================================================================
        // Cross
        // =========================================================================

        TEST_METHOD(Cross) {
            constexpr VecSIMD<float, 3> a{ 1, 0, 0 };
            constexpr VecSIMD<float, 3> b{ 0, 1, 0 };

            const auto result = ::Cross(a, b);

            Assert::AreEqual(0.0f, result[0]);
            Assert::AreEqual(0.0f, result[1]);
            Assert::AreEqual(1.0f, result[2]);
        }

        TEST_METHOD(CrossOfParallelVecSIMDtorsIsZero) {
            constexpr VecSIMD<float, 3> a{ 2, 4, 6 };
            constexpr VecSIMD<float, 3> b{ 1, 2, 3 };

            constexpr auto result = ::Cross(a, b);

            Assert::AreEqual(0.0f, result[0]);
            Assert::AreEqual(0.0f, result[1]);
            Assert::AreEqual(0.0f, result[2]);
        }

        TEST_METHOD(CrossIsAnticommutative) {
            constexpr VecSIMD<float, 3> a{ 1, 2, 3 };
            constexpr VecSIMD<float, 3> b{ 4, 5, 6 };

            constexpr auto ab = ::Cross(a, b);
            constexpr auto ba = ::Cross(b, a);

            Assert::IsTrue(ab[0] == -ba[0]);
            Assert::IsTrue(ab[1] == -ba[1]);
            Assert::IsTrue(ab[2] == -ba[2]);
        }

        // =========================================================================
        // Distance
        // =========================================================================

        TEST_METHOD(Distance) {
            constexpr VecSIMD<float, 3> a{ 0.0f, 0.0f, 0.0f };
            constexpr VecSIMD<float, 3> b{ 3.0f, 4.0f, 0.0f };

            ExpectNear(5.0f, ::Distance(a, b));
        }

        TEST_METHOD(DistanceToSelfIsZero) {
            constexpr VecSIMD<float, 3> a{ 1.0f, 2.0f, 3.0f };

            ExpectNear(0.0f, ::Distance(a, a));
        }

        TEST_METHOD(DistanceIsSymmetric) {
            constexpr VecSIMD<float, 3> a{ 1.0f, 2.0f, 3.0f };
            constexpr VecSIMD<float, 3> b{ 4.0f, 6.0f, 3.0f };

            ExpectNear(::Distance(b, a), ::Distance(a, b));
        }

        // =========================================================================
        // Angle
        // =========================================================================

        TEST_METHOD(AngleBetweenIdenticalVecSIMDtors) {
            constexpr VecSIMD<float, 3> a{ 1.0f, 0.0f, 0.0f };

            ExpectNear(0.0f, Angle(a, a), 1e-5f);
        }

        TEST_METHOD(AngleBetweenPerpendicularVecSIMDtors) {
            constexpr VecSIMD<float, 3> a{ 1.0f, 0.0f, 0.0f };
            constexpr VecSIMD<float, 3> b{ 0.0f, 1.0f, 0.0f };

            ExpectNear(std::numbers::pi_v<float> / 2.0f, Angle(a, b), 1e-5f);
        }

        TEST_METHOD(AngleBetweenOpposingVecSIMDtors) {
            constexpr VecSIMD<float, 3> a{ 1.0f, 0.0f, 0.0f };
            constexpr VecSIMD<float, 3> b{ -1.0f, 0.0f, 0.0f };

            ExpectNear(std::numbers::pi_v<float>, Angle(a, b), 1e-5f);
        }

        TEST_METHOD(AngleIsScaleInvariant) {
            constexpr VecSIMD<float, 3> a{ 1.0f, 0.0f, 0.0f };
            constexpr VecSIMD<float, 3> b{ 0.0f, 5.0f, 0.0f };

            ExpectNear(std::numbers::pi_v<float> / 2.0f, Angle(a, b), 1e-5f);
        }

        // =========================================================================
        // Lerp
        // =========================================================================

        TEST_METHOD(LerpAtStart) {
            constexpr VecSIMD<float, 3> a{ 0.0f, 0.0f, 0.0f };
            constexpr VecSIMD<float, 3> b{ 10.0f, 20.0f, 30.0f };

            const auto result = Lerp(a, b, 0.0f);

            ExpectNear(0.0f, result[0]);
            ExpectNear(0.0f, result[1]);
            ExpectNear(0.0f, result[2]);
        }

        TEST_METHOD(LerpAtEnd) {
            constexpr VecSIMD<float, 3> a{ 0.0f, 0.0f, 0.0f };
            constexpr VecSIMD<float, 3> b{ 10.0f, 20.0f, 30.0f };

            const auto result = Lerp(a, b, 1.0f);

            ExpectNear(10.0f, result[0]);
            ExpectNear(20.0f, result[1]);
            ExpectNear(30.0f, result[2]);
        }

        TEST_METHOD(LerpAtMidpoint) {
            constexpr VecSIMD<float, 3> a{ 0.0f, 0.0f, 0.0f };
            constexpr VecSIMD<float, 3> b{ 10.0f, 20.0f, 30.0f };

            const auto result = Lerp(a, b, 0.5f);

            ExpectNear(5.0f, result[0]);
            ExpectNear(10.0f, result[1]);
            ExpectNear(15.0f, result[2]);
        }

        // =========================================================================
        // Scale
        // =========================================================================

        TEST_METHOD(Scale) {
            constexpr VecSIMD<float, 3> a{ 2, 3, 4 };
            constexpr VecSIMD<float, 3> b{ 5, 6, 7 };

            const auto result = ::Scale(a, b);

            Assert::AreEqual(10.0f, result[0]);
            Assert::AreEqual(18.0f, result[1]);
            Assert::AreEqual(28.0f, result[2]);
        }

        TEST_METHOD(ScaleMatchesComponentMultiplication) {
            constexpr VecSIMD<float, 3> a{ 2, 3, 4 };
            constexpr VecSIMD<float, 3> b{ 5, 6, 7 };

            Assert::IsTrue(::Scale(a, b) == (a * b));
        }

        // =========================================================================
        // Min / Max
        // =========================================================================

        TEST_METHOD(Min) {
            constexpr VecSIMD<float, 3> a{ 1, 5, 3 };
            constexpr VecSIMD<float, 3> b{ 4, 2, 6 };

            const auto result = ::Min(a, b);

            Assert::AreEqual(1.0f, result[0]);
            Assert::AreEqual(2.0f, result[1]);
            Assert::AreEqual(3.0f, result[2]);
        }

        TEST_METHOD(Max) {
            constexpr VecSIMD<float, 3> a{ 1, 5, 3 };
            constexpr VecSIMD<float, 3> b{ 4, 2, 6 };

            const auto result = ::Max(a, b);

            Assert::AreEqual(4.0f, result[0]);
            Assert::AreEqual(5.0f, result[1]);
            Assert::AreEqual(6.0f, result[2]);
        }

        // =========================================================================
        // MoveTowards
        // =========================================================================

        TEST_METHOD(MoveTowardsPartialStep) {
            constexpr VecSIMD<float, 3> current{ 0.0f, 0.0f, 0.0f };
            constexpr VecSIMD<float, 3> target{ 10.0f, 0.0f, 0.0f };

            const auto result = MoveTowards(current, target, 4.0f);

            ExpectNear(4.0f, result[0]);
            ExpectNear(0.0f, result[1]);
            ExpectNear(0.0f, result[2]);
        }

        TEST_METHOD(MoveTowardsOvershootClampsToTarget) {
            constexpr VecSIMD<float, 3> current{ 0.0f, 0.0f, 0.0f };
            constexpr VecSIMD<float, 3> target{ 10.0f, 0.0f, 0.0f };

            const auto result = MoveTowards(current, target, 100.0f);

            ExpectNear(10.0f, result[0]);
            ExpectNear(0.0f, result[1]);
            ExpectNear(0.0f, result[2]);
        }

        TEST_METHOD(MoveTowardsSamePointReturnsTarget) {
            constexpr VecSIMD<float, 3> current{ 5.0f, 5.0f, 5.0f };
            constexpr VecSIMD<float, 3> target{ 5.0f, 5.0f, 5.0f };

            const auto result = MoveTowards(current, target, 1.0f);

            ExpectNear(5.0f, result[0]);
            ExpectNear(5.0f, result[1]);
            ExpectNear(5.0f, result[2]);
        }

        // =========================================================================
        // Reflect
        // =========================================================================

        TEST_METHOD(ReflectOffFlatSurface) {
            constexpr VecSIMD<float, 3> v{ 1.0f, -1.0f, 0.0f };
            constexpr VecSIMD<float, 3> normal{ 0.0f, 1.0f, 0.0f };

            const auto result = Reflect(v, normal);

            ExpectNear(1.0f, result[0]);
            ExpectNear(1.0f, result[1]);
            ExpectNear(0.0f, result[2]);
        }

        TEST_METHOD(ReflectStraightOnBouncesBack) {
            constexpr VecSIMD<float, 3> v{ 0.0f, -1.0f, 0.0f };
            constexpr VecSIMD<float, 3> normal{ 0.0f, 1.0f, 0.0f };

            const auto result = Reflect(v, normal);

            ExpectNear(0.0f, result[0]);
            ExpectNear(1.0f, result[1]);
            ExpectNear(0.0f, result[2]);
        }

        // =========================================================================
        // Perpendicular
        // =========================================================================

        TEST_METHOD(Perpendicular) {
            constexpr VecSIMD<float, 2> v{ 1.0f, 0.0f };

            const auto result = ::Perpendicular(v);

            ExpectNear(0.0f, result[0]);
            ExpectNear(1.0f, result[1]);
        }

        TEST_METHOD(PerpendicularIsOrthogonal) {
            constexpr VecSIMD<float, 2> v{ 3.0f, 4.0f };

            const auto result = ::Perpendicular(v);

            ExpectNear(0.0f, ::Dot(v, result));
        }

        TEST_METHOD(PerpendicularPreservesLength) {
            const VecSIMD<float, 2> v{ 3.0f, 4.0f };

            const auto result = ::Perpendicular(v);

            ExpectNear(v.Length(), result.Length());
        }

        // =========================================================================
        // Higher dimensions
        // =========================================================================

        TEST_METHOD(HigherDimension) {
            VecSIMD<float, 8> vec{ 1, 2, 3, 4, 5, 6, 7, 8 };

            for (std::size_t i = 0; i < 8; ++i) {
                Assert::AreEqual(static_cast<float>(i + 1), vec[i]);
            }
        }

        // =========================================================================
        // Formatting
        // =========================================================================

        TEST_METHOD(Formatting) {
            constexpr VecSIMD<float, 3> vec{ 1, 2, 3 };

            Assert::AreEqual(std::string("(1, 2, 3)"), std::format("{}", vec));
        }

        TEST_METHOD(FormattingFloatingPoint) {
            constexpr VecSIMD<float, 4> vec{ 1.0f, 2.5f, 3.25f, 4.0f };

            Assert::AreEqual(std::string("(1, 2.5, 3.25, 4)"), std::format("{}", vec));
        }

        // =========================================================================
// 4D vectors
// =========================================================================

        TEST_METHOD(Addition4D) {
            constexpr VecSIMD<float, 4> a{ 1, 2, 3, 4 };
            constexpr VecSIMD<float, 4> b{ 5, 6, 7, 8 };

            const auto result = a + b;

            Assert::AreEqual(6.0f, result[0]);
            Assert::AreEqual(8.0f, result[1]);
            Assert::AreEqual(10.0f, result[2]);
            Assert::AreEqual(12.0f, result[3]);
        }

        TEST_METHOD(Subtraction4D) {
            constexpr VecSIMD<float, 4> a{ 5, 7, 9, 11 };
            constexpr VecSIMD<float, 4> b{ 1, 2, 3, 4 };

            const auto result = a - b;

            Assert::AreEqual(4.0f, result[0]);
            Assert::AreEqual(5.0f, result[1]);
            Assert::AreEqual(6.0f, result[2]);
            Assert::AreEqual(7.0f, result[3]);
        }

        TEST_METHOD(ComponentMultiplication4D) {
            constexpr VecSIMD<float, 4> a{ 1, 2, 3, 4 };
            constexpr VecSIMD<float, 4> b{ 2, 3, 4, 5 };

            const auto result = a * b;

            Assert::AreEqual(2.0f, result[0]);
            Assert::AreEqual(6.0f, result[1]);
            Assert::AreEqual(12.0f, result[2]);
            Assert::AreEqual(20.0f, result[3]);
        }

        TEST_METHOD(ComponentDivision4D) {
            constexpr VecSIMD<float, 4> a{ 10, 20, 30, 40 };
            constexpr VecSIMD<float, 4> b{ 2, 4, 5, 8 };

            const auto result = a / b;

            Assert::AreEqual(5.0f, result[0]);
            Assert::AreEqual(5.0f, result[1]);
            Assert::AreEqual(6.0f, result[2]);
            Assert::AreEqual(5.0f, result[3]);
        }

        TEST_METHOD(ScalarMultiplication4D) {
            constexpr VecSIMD<float, 4> vec{ 1, 2, 3, 4 };

            const auto result = vec * 2.0f;

            Assert::AreEqual(2.0f, result[0]);
            Assert::AreEqual(4.0f, result[1]);
            Assert::AreEqual(6.0f, result[2]);
            Assert::AreEqual(8.0f, result[3]);
        }

        TEST_METHOD(ScalarDivision4D) {
            constexpr VecSIMD<float, 4> vec{ 10, 20, 30, 40 };

            const auto result = vec / 2.0f;

            Assert::AreEqual(5.0f, result[0]);
            Assert::AreEqual(10.0f, result[1]);
            Assert::AreEqual(15.0f, result[2]);
            Assert::AreEqual(20.0f, result[3]);
        }

        TEST_METHOD(Dot4D) {
            constexpr VecSIMD<float, 4> a{ 1, 2, 3, 4 };
            constexpr VecSIMD<float, 4> b{ 5, 6, 7, 8 };

            Assert::AreEqual(70.0f, ::Dot(a, b));
        }

        TEST_METHOD(LengthSquared4D) {
            constexpr VecSIMD<float, 4> vec{ 1, 2, 3, 4 };

            Assert::AreEqual(30.0f, vec.LengthSquared());
        }

        TEST_METHOD(Length4D) {
            const VecSIMD<float, 4> vec{ 0, 0, 0, 5 };

            ExpectNear(5.0f, vec.Length());
        }

        TEST_METHOD(Normalized4D) {
            const VecSIMD<float, 4> vec{ 0, 0, 0, 5 };

            const auto result = vec.Normalized();

            ExpectNear(1.0f, result.Length());
            ExpectNear(0.0f, result[0]);
            ExpectNear(0.0f, result[1]);
            ExpectNear(0.0f, result[2]);
            ExpectNear(1.0f, result[3]);
        }

        TEST_METHOD(Lerp4D) {
            constexpr VecSIMD<float, 4> a{ 0, 0, 0, 0 };
            constexpr VecSIMD<float, 4> b{ 10, 20, 30, 40 };

            const auto result = Lerp(a, b, 0.5f);

            ExpectNear(5.0f, result[0]);
            ExpectNear(10.0f, result[1]);
            ExpectNear(15.0f, result[2]);
            ExpectNear(20.0f, result[3]);
        }

        TEST_METHOD(Scale4D) {
            constexpr VecSIMD<float, 4> a{ 2, 3, 4, 5 };
            constexpr VecSIMD<float, 4> b{ 5, 6, 7, 8 };

            const auto result = ::Scale(a, b);

            Assert::AreEqual(10.0f, result[0]);
            Assert::AreEqual(18.0f, result[1]);
            Assert::AreEqual(28.0f, result[2]);
            Assert::AreEqual(40.0f, result[3]);
        }

        TEST_METHOD(Min4D) {
            constexpr VecSIMD<float, 4> a{ 1, 5, 3, 8 };
            constexpr VecSIMD<float, 4> b{ 4, 2, 6, 7 };

            const auto result = ::Min(a, b);

            Assert::AreEqual(1.0f, result[0]);
            Assert::AreEqual(2.0f, result[1]);
            Assert::AreEqual(3.0f, result[2]);
            Assert::AreEqual(7.0f, result[3]);
        }

        TEST_METHOD(Max4D) {
            constexpr VecSIMD<float, 4> a{ 1, 5, 3, 8 };
            constexpr VecSIMD<float, 4> b{ 4, 2, 6, 7 };

            const auto result = ::Max(a, b);

            Assert::AreEqual(4.0f, result[0]);
            Assert::AreEqual(5.0f, result[1]);
            Assert::AreEqual(6.0f, result[2]);
            Assert::AreEqual(8.0f, result[3]);
        }

        TEST_METHOD(MoveTowards4D) {
            constexpr VecSIMD<float, 4> current{ 0, 0, 0, 0 };
            constexpr VecSIMD<float, 4> target{ 10, 0, 0, 0 };

            const auto result = MoveTowards(current, target, 4.0f);

            ExpectNear(4.0f, result[0]);
            ExpectNear(0.0f, result[1]);
            ExpectNear(0.0f, result[2]);
            ExpectNear(0.0f, result[3]);
        }

        TEST_METHOD(Reflect4D) {
            constexpr VecSIMD<float, 4> v{ 1, -1, 0, 0 };
            constexpr VecSIMD<float, 4> normal{ 0, 1, 0, 0 };

            const auto result = Reflect(v, normal);

            ExpectNear(1.0f, result[0]);
            ExpectNear(1.0f, result[1]);
            ExpectNear(0.0f, result[2]);
            ExpectNear(0.0f, result[3]);
        }

        TEST_METHOD(Distance4D) {
            constexpr VecSIMD<float, 4> a{ 0, 0, 0, 0 };
            constexpr VecSIMD<float, 4> b{ 3, 4, 0, 0 };

            ExpectNear(5.0f, ::Distance(a, b));
        }

        TEST_METHOD(Angle4D) {
            constexpr VecSIMD<float, 4> a{ 1, 0, 0, 0 };
            constexpr VecSIMD<float, 4> b{ 0, 1, 0, 0 };

            ExpectNear(
                std::numbers::pi_v<float> / 2.0f,
                Angle(a, b),
                1e-5f
            );
        }
    };
}
