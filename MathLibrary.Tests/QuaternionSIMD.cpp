#include "pch.h"
#include "CppUnitTest.h"

#include "QuaternionSIMD.h"
#include "VecSIMD.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace {
    constexpr float kPi = std::numbers::pi_v<float>;

    // Compares two floating-point values through Assert::AreEqual's tolerance overload.
    void ExpectNear(double expected, double actual, double tolerance = 1e-4) {
        Assert::AreEqual(expected, actual, tolerance);
    }

    template <float_num T>
    void ExpectNear(const QuaternionSIMD<T>& a, const QuaternionSIMD<T>& b, T tolerance = static_cast<T>(1e-4)) {
        ExpectNear(a.X(), b.X(), tolerance);
        ExpectNear(a.Y(), b.Y(), tolerance);
        ExpectNear(a.Z(), b.Z(), tolerance);
        ExpectNear(a.W(), b.W(), tolerance);
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
    TEST_CLASS(QuaternionSIMDTests)
    {
    public:
        // =========================================================================
        // Construction
        // =========================================================================

        TEST_METHOD(DefaultConstruction) {
            constexpr QuaternionSIMD<float> q{};

            ExpectNear(0.0f, q.X());
            ExpectNear(0.0f, q.Y());
            ExpectNear(0.0f, q.Z());
            ExpectNear(1.0f, q.W());
        }

        TEST_METHOD(ValueConstruction) {
            constexpr QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            ExpectNear(1.0f, q.X());
            ExpectNear(2.0f, q.Y());
            ExpectNear(3.0f, q.Z());
            ExpectNear(4.0f, q.W());
        }

        TEST_METHOD(Identity) {
            constexpr QuaternionSIMD<float> q = QuaternionSIMD<float>::Identity();

            ExpectNear(0.0f, q.X());
            ExpectNear(0.0f, q.Y());
            ExpectNear(0.0f, q.Z());
            ExpectNear(1.0f, q.W());
        }

        TEST_METHOD(FromAxisAngle) {
            constexpr VecSIMD<float, 3> axis{ 0.0f, 1.0f, 0.0f };
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromAxisAngle(axis, kPi / 2.0f);

            ExpectNear(0.0f, q.X(), 1e-4f);
            ExpectNear(std::sin(kPi / 4.0f), q.Y(), 1e-4f);
            ExpectNear(0.0f, q.Z(), 1e-4f);
            ExpectNear(std::cos(kPi / 4.0f), q.W(), 1e-4f);
        }

        TEST_METHOD(FromAxisAngleNormalizesAxis) {
            constexpr VecSIMD<float, 3> axis{ 0.0f, 5.0f, 0.0f };
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromAxisAngle(axis, kPi / 2.0f);

            ExpectNear(1.0f, q.Length(), 1e-4f);
            ExpectNear(std::sin(kPi / 4.0f), q.Y(), 1e-4f);
        }

        TEST_METHOD(FromAxisAngleZeroAxis) {
            constexpr VecSIMD<float, 3> axis{ 0.0f, 0.0f, 0.0f };
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromAxisAngle(axis, kPi / 2.0f);

            ExpectNear(0.0f, q.X(), 1e-4f);
            ExpectNear(0.0f, q.Y(), 1e-4f);
            ExpectNear(0.0f, q.Z(), 1e-4f);
        }

        TEST_METHOD(AxisAngleConstructor) {
            constexpr VecSIMD<float, 3> axis{ 0.0f, 0.0f, 1.0f };
            const QuaternionSIMD<float> q{ axis, kPi };

            ExpectNear(q, QuaternionSIMD<float>::FromAxisAngle(axis, kPi));
        }

        TEST_METHOD(FromEulerIdentity) {
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromEuler(0.0f, 0.0f, 0.0f);

            ExpectNear(q, QuaternionSIMD<float>::Identity());
        }

        TEST_METHOD(FromEulerMatchesAxisAngleForSingleAxis) {
            constexpr float angle = kPi / 3.0f;

            const QuaternionSIMD<float> fromEuler = QuaternionSIMD<float>::FromEuler(angle, 0.0f, 0.0f);
            const QuaternionSIMD<float> fromAxisAngle =
                QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{1.0f, 0.0f, 0.0f}, angle);

            ExpectNear(fromEuler, fromAxisAngle);
        }

        TEST_METHOD(MixedValueTypes) {
            constexpr QuaternionSIMD<double> q{ 1, 2.5, 3, 4 };

            ExpectNear(1.0, q.X(), 1e-9);
            ExpectNear(2.5, q.Y(), 1e-9);
            ExpectNear(3.0, q.Z(), 1e-9);
            ExpectNear(4.0, q.W(), 1e-9);
        }

        // =========================================================================
        // FromToRotation
        // =========================================================================

        TEST_METHOD(FromToRotationIdenticalVectors) {
            constexpr VecSIMD<float, 3> v{ 1.0f, 0.0f, 0.0f };

            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromToRotation(v, v);

            ExpectNear(q, QuaternionSIMD<float>::Identity());
        }

        TEST_METHOD(FromToRotationOpposingVectors) {
            constexpr VecSIMD<float, 3> from{ 1.0f, 0.0f, 0.0f };
            constexpr VecSIMD<float, 3> to{ -1.0f, 0.0f, 0.0f };

            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromToRotation(from, to);

            ExpectNear(1.0f, q.Length(), 1e-4f);

            const VecSIMD<float, 3> rotated = q.RotateVector(from);

            ExpectNear(rotated, to);
        }

        TEST_METHOD(FromToRotationRotatesFromOntoTo) {
            constexpr VecSIMD<float, 3> from{ 1.0f, 0.0f, 0.0f };
            constexpr VecSIMD<float, 3> to{ 0.0f, 1.0f, 0.0f };

            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromToRotation(from, to);
            const VecSIMD<float, 3> rotated = q.RotateVector(from);

            ExpectNear(rotated, to);
        }

        TEST_METHOD(FromToRotationIsNormalized) {
            constexpr VecSIMD<float, 3> from{ 1.0f, 0.0f, 0.0f };
            constexpr VecSIMD<float, 3> to{ 0.0f, 0.0f, 1.0f };

            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromToRotation(from, to);

            ExpectNear(1.0f, q.Length(), 1e-4f);
        }

        // =========================================================================
        // LookRotation
        // =========================================================================

        TEST_METHOD(LookRotationIdentityForForwardZ) {
            constexpr VecSIMD<float, 3> forward{ 0.0f, 0.0f, 1.0f };
            constexpr VecSIMD<float, 3> up{ 0.0f, 1.0f, 0.0f };

            const QuaternionSIMD<float> q = QuaternionSIMD<float>::LookRotation(forward, up);

            ExpectNear(q, QuaternionSIMD<float>::Identity());
        }

        TEST_METHOD(LookRotationRotatesForwardVector) {
            constexpr VecSIMD<float, 3> forward{ 1.0f, 0.0f, 0.0f };
            constexpr VecSIMD<float, 3> up{ 0.0f, 1.0f, 0.0f };

            const QuaternionSIMD<float> q = QuaternionSIMD<float>::LookRotation(forward, up);
            const VecSIMD<float, 3> rotated = q.RotateVector(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f});

            ExpectNear(rotated, forward);
        }

        TEST_METHOD(LookRotationIsNormalized) {
            constexpr VecSIMD<float, 3> forward{ 0.0f, 0.0f, 1.0f };
            constexpr VecSIMD<float, 3> up{ 0.0f, 1.0f, 0.0f };

            const QuaternionSIMD<float> q = QuaternionSIMD<float>::LookRotation(forward, up);

            ExpectNear(1.0f, q.Length(), 1e-4f);
        }

        // =========================================================================
        // Access
        // =========================================================================

        TEST_METHOD(ComponentAccess) {
            QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            q.X() = 10.0f;
            q.Y() = 20.0f;
            q.Z() = 30.0f;
            q.W() = 40.0f;

            ExpectNear(10.0f, q.X());
            ExpectNear(20.0f, q.Y());
            ExpectNear(30.0f, q.Z());
            ExpectNear(40.0f, q.W());
        }

        TEST_METHOD(ConstComponentAccess) {
            constexpr QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            ExpectNear(1.0f, q.X());
            ExpectNear(2.0f, q.Y());
            ExpectNear(3.0f, q.Z());
            ExpectNear(4.0f, q.W());
        }

        TEST_METHOD(Data) {
            QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            float* data = q.Data();

            Assert::IsNotNull(data);

            data[0] = 10.0f;
            data[1] = 20.0f;
            data[2] = 30.0f;
            data[3] = 40.0f;

            ExpectNear(10.0f, q.X());
            ExpectNear(20.0f, q.Y());
            ExpectNear(30.0f, q.Z());
            ExpectNear(40.0f, q.W());
        }

        TEST_METHOD(Size) {
            Assert::AreEqual(static_cast<std::size_t>(4), static_cast<std::size_t>((QuaternionSIMD<float>::Size())));
            Assert::AreEqual(static_cast<std::size_t>(4), static_cast<std::size_t>((QuaternionSIMD<double>::Size())));
        }

        // =========================================================================
        // Equality
        // =========================================================================

        TEST_METHOD(Equality) {
            constexpr QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr QuaternionSIMD<float> b{ 1.0f, 2.0f, 3.0f, 4.0f };

            Assert::IsTrue(a == b);
        }

        TEST_METHOD(Inequality) {
            constexpr QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr QuaternionSIMD<float> b{ 1.0f, 2.0f, 3.0f, 5.0f };

            Assert::IsFalse(a == b);
        }

        TEST_METHOD(EqualityDifferentTypes) {
            constexpr QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr QuaternionSIMD<double> b{ 1.0, 2.0, 3.0, 4.0 };

            Assert::IsTrue(a == b);
        }

        // =========================================================================
        // Addition
        // =========================================================================

        TEST_METHOD(Addition) {
            constexpr QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr QuaternionSIMD<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

            const auto result = a + b;

            ExpectNear(5.0f, result.X());
            ExpectNear(7.0f, result.Y());
            ExpectNear(9.0f, result.Z());
            ExpectNear(11.0f, result.W());
        }

        TEST_METHOD(AdditionCommonType) {
            constexpr QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr QuaternionSIMD<double> b{ 0.5, 1.5, 2.5, 3.5 };

            const auto result = a + b;

            static_assert(std::is_same_v<decltype(result), const QuaternionSIMD<double>>);

            ExpectNear(1.5, result.X(), 1e-9);
            ExpectNear(3.5, result.Y(), 1e-9);
            ExpectNear(5.5, result.Z(), 1e-9);
            ExpectNear(7.5, result.W(), 1e-9);
        }

        TEST_METHOD(AdditionAssignment) {
            QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr QuaternionSIMD<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

            a += b;

            ExpectNear(5.0f, a.X());
            ExpectNear(7.0f, a.Y());
            ExpectNear(9.0f, a.Z());
            ExpectNear(11.0f, a.W());
        }

        TEST_METHOD(MixedTypeAdditionAssignment) {
            QuaternionSIMD<double> a{ 1.0, 2.0, 3.0, 4.0 };
            constexpr QuaternionSIMD<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

            a += b;

            ExpectNear(5.0, a.X(), 1e-9);
            ExpectNear(7.0, a.Y(), 1e-9);
            ExpectNear(9.0, a.Z(), 1e-9);
            ExpectNear(11.0, a.W(), 1e-9);
        }

        // =========================================================================
        // Subtraction
        // =========================================================================

        TEST_METHOD(Subtraction) {
            constexpr QuaternionSIMD<float> a{ 5.0f, 7.0f, 9.0f, 11.0f };
            constexpr QuaternionSIMD<float> b{ 1.0f, 2.0f, 3.0f, 4.0f };

            const auto result = a - b;

            ExpectNear(4.0f, result.X());
            ExpectNear(5.0f, result.Y());
            ExpectNear(6.0f, result.Z());
            ExpectNear(7.0f, result.W());
        }

        TEST_METHOD(SubtractionCommonType) {
            constexpr QuaternionSIMD<float> a{ 5.0f, 7.0f, 9.0f, 11.0f };
            constexpr QuaternionSIMD<double> b{ 0.5, 1.5, 2.5, 3.5 };

            const auto result = a - b;

            static_assert(std::is_same_v<decltype(result), const QuaternionSIMD<double>>);

            ExpectNear(4.5, result.X(), 1e-9);
            ExpectNear(5.5, result.Y(), 1e-9);
            ExpectNear(6.5, result.Z(), 1e-9);
            ExpectNear(7.5, result.W(), 1e-9);
        }

        TEST_METHOD(SubtractionAssignment) {
            QuaternionSIMD<float> a{ 5.0f, 7.0f, 9.0f, 11.0f };
            constexpr QuaternionSIMD<float> b{ 1.0f, 2.0f, 3.0f, 4.0f };

            a -= b;

            ExpectNear(4.0f, a.X());
            ExpectNear(5.0f, a.Y());
            ExpectNear(6.0f, a.Z());
            ExpectNear(7.0f, a.W());
        }

        // =========================================================================
        // Hamilton product
        // =========================================================================

        TEST_METHOD(MultiplicationByIdentity) {
            constexpr QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr QuaternionSIMD<float> identity = QuaternionSIMD<float>::Identity();

            const auto result = q * identity;

            ExpectNear(q.X(), result.X());
            ExpectNear(q.Y(), result.Y());
            ExpectNear(q.Z(), result.Z());
            ExpectNear(q.W(), result.W());
        }

        TEST_METHOD(MultiplicationIsNonCommutative) {
            const QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{1.0f, 0.0f, 0.0f}, kPi / 2.0f);
            const QuaternionSIMD<float> b = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);

            const QuaternionSIMD<float> ab = a * b;
            const QuaternionSIMD<float> ba = b * a;

            Assert::IsFalse((ab.X() == ba.X()) && (ab.Y() == ba.Y()) && (ab.Z() == ba.Z()) && (ab.W() == ba.W()));
        }

        TEST_METHOD(MultiplicationCombinesRotations) {
            // Two 90 degree rotations around Z should equal one 180 degree rotation around Z.
            const QuaternionSIMD<float> quarterTurn =
                QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);
            const QuaternionSIMD<float> halfTurn = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi);

            const QuaternionSIMD<float> combined = quarterTurn * quarterTurn;

            ExpectNear(combined, halfTurn);
        }

        TEST_METHOD(MultiplicationCommonType) {
            constexpr QuaternionSIMD<float> a{ 0.0f, 0.0f, 0.0f, 1.0f };
            constexpr QuaternionSIMD<double> b{ 1.0, 0.0, 0.0, 0.0 };

            const auto result = a * b;

            static_assert(std::is_same_v<decltype(result), const QuaternionSIMD<double>>);

            ExpectNear(1.0, result.X(), 1e-9);
            ExpectNear(0.0, result.Y(), 1e-9);
            ExpectNear(0.0, result.Z(), 1e-9);
            ExpectNear(0.0, result.W(), 1e-9);
        }

        TEST_METHOD(MultiplicationAssignment) {
            QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);
            const QuaternionSIMD<float> b = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);

            a *= b;

            ExpectNear(a, QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi));
        }

        // =========================================================================
        // Scalar multiplication
        // =========================================================================

        TEST_METHOD(ScalarMultiplication) {
            constexpr QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            const auto result = q * 2.0f;

            ExpectNear(2.0f, result.X());
            ExpectNear(4.0f, result.Y());
            ExpectNear(6.0f, result.Z());
            ExpectNear(8.0f, result.W());
        }

        TEST_METHOD(ScalarMultiplicationCommonType) {
            constexpr QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            const auto result = q * 0.5;

            static_assert(std::is_same_v<decltype(result), const QuaternionSIMD<double>>);

            ExpectNear(0.5, result.X(), 1e-9);
            ExpectNear(1.0, result.Y(), 1e-9);
            ExpectNear(1.5, result.Z(), 1e-9);
            ExpectNear(2.0, result.W(), 1e-9);
        }

        TEST_METHOD(ScalarMultiplicationReversed) {
            constexpr QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            const auto result = 2.0f * q;

            ExpectNear(2.0f, result.X());
            ExpectNear(4.0f, result.Y());
            ExpectNear(6.0f, result.Z());
            ExpectNear(8.0f, result.W());
        }

        TEST_METHOD(ScalarMultiplicationAssignment) {
            QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            q *= 3.0f;

            ExpectNear(3.0f, q.X());
            ExpectNear(6.0f, q.Y());
            ExpectNear(9.0f, q.Z());
            ExpectNear(12.0f, q.W());
        }

        // =========================================================================
        // Scalar division
        // =========================================================================

        TEST_METHOD(ScalarDivision) {
            constexpr QuaternionSIMD<float> q{ 2.0f, 4.0f, 6.0f, 8.0f };

            const auto result = q / 2.0f;

            ExpectNear(1.0f, result.X());
            ExpectNear(2.0f, result.Y());
            ExpectNear(3.0f, result.Z());
            ExpectNear(4.0f, result.W());
        }

        TEST_METHOD(ScalarDivisionCommonType) {
            constexpr QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            const auto result = q / 2.0;

            static_assert(std::is_same_v<decltype(result), const QuaternionSIMD<double>>);

            ExpectNear(0.5, result.X(), 1e-9);
            ExpectNear(1.0, result.Y(), 1e-9);
            ExpectNear(1.5, result.Z(), 1e-9);
            ExpectNear(2.0, result.W(), 1e-9);
        }

        TEST_METHOD(ScalarDivisionAssignment) {
            QuaternionSIMD<float> q{ 2.0f, 4.0f, 6.0f, 8.0f };

            q /= 2.0f;

            ExpectNear(1.0f, q.X());
            ExpectNear(2.0f, q.Y());
            ExpectNear(3.0f, q.Z());
            ExpectNear(4.0f, q.W());
        }

        // =========================================================================
        // Unary operators
        // =========================================================================

        TEST_METHOD(UnaryPlus) {
            constexpr QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            const auto result = +q;

            ExpectNear(1.0f, result.X());
            ExpectNear(2.0f, result.Y());
            ExpectNear(3.0f, result.Z());
            ExpectNear(4.0f, result.W());
        }

        TEST_METHOD(UnaryMinus) {
            constexpr QuaternionSIMD<float> q{ 1.0f, -2.0f, 3.0f, -4.0f };

            const auto result = -q;

            ExpectNear(-1.0f, result.X());
            ExpectNear(2.0f, result.Y());
            ExpectNear(-3.0f, result.Z());
            ExpectNear(4.0f, result.W());
        }

        // =========================================================================
        // Length / normalization
        // =========================================================================

        TEST_METHOD(LengthSquared) {
            constexpr QuaternionSIMD<float> q{ 1.0f, 2.0f, 2.0f, 4.0f };

            ExpectNear(25.0f, q.LengthSquared());
        }

        TEST_METHOD(Length) {
            constexpr QuaternionSIMD<float> q{ 0.0f, 0.0f, 0.0f, 3.0f };

            ExpectNear(3.0f, q.Length());
        }

        TEST_METHOD(IdentityLengthIsOne) {
            constexpr QuaternionSIMD<float> q = QuaternionSIMD<float>::Identity();

            ExpectNear(1.0f, q.Length());
        }

        TEST_METHOD(Normalized) {
            constexpr QuaternionSIMD<float> q{ 0.0f, 0.0f, 0.0f, 2.0f };

            const QuaternionSIMD<float> result = q.Normalized();

            ExpectNear(1.0f, result.Length());
            ExpectNear(1.0f, result.W());
        }

        TEST_METHOD(NormalizedDoesNotMutate) {
            constexpr QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            const QuaternionSIMD<float> result = q.Normalized();

            ExpectNear(1.0f, q.X());
            ExpectNear(2.0f, q.Y());
            ExpectNear(3.0f, q.Z());
            ExpectNear(4.0f, q.W());
            ExpectNear(1.0f, result.Length(), 1e-4f);
        }

        TEST_METHOD(NormalizedZeroQuaternionSIMD) {
            constexpr QuaternionSIMD<float> q{ 0.0f, 0.0f, 0.0f, 0.0f };

            const QuaternionSIMD<float> result = q.Normalized();

            ExpectNear(0.0f, result.X());
            ExpectNear(0.0f, result.Y());
            ExpectNear(0.0f, result.Z());
            ExpectNear(0.0f, result.W());
        }

        TEST_METHOD(Normalize) {
            QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            q.Normalize();

            ExpectNear(1.0f, q.Length(), 1e-4f);
        }

        // =========================================================================
        // Conjugate / inverse
        // =========================================================================

        TEST_METHOD(Conjugate) {
            constexpr QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            const auto result = q.Conjugate();

            ExpectNear(-1.0f, result.X());
            ExpectNear(-2.0f, result.Y());
            ExpectNear(-3.0f, result.Z());
            ExpectNear(4.0f, result.W());
        }

        TEST_METHOD(InverseOfUnitQuaternionSIMDEqualsConjugate) {
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, kPi / 3.0f);

            ExpectNear(q.Inverse(), q.Conjugate());
        }

        TEST_METHOD(InverseTimesSelfIsIdentity) {
            const QuaternionSIMD<float> q = {1.0f, 1.0f, 0.0f, 0.0f};

            const QuaternionSIMD<float> result = q * q.Inverse();

            ExpectNear(result, QuaternionSIMD<float>::Identity());
        }

        TEST_METHOD(InverseOfZeroQuaternionSIMDReturnsConjugate) {
            constexpr QuaternionSIMD<float> q{ 0.0f, 0.0f, 0.0f, 0.0f };

            const auto result = q.Inverse();

            ExpectNear(0.0f, result.X());
            ExpectNear(0.0f, result.Y());
            ExpectNear(0.0f, result.Z());
            ExpectNear(0.0f, result.W());
        }

        TEST_METHOD(InverseOfNonUnitQuaternionSIMD) {
            constexpr QuaternionSIMD<float> q{ 0.0f, 0.0f, 0.0f, 2.0f };

            const QuaternionSIMD<float> result = q.Inverse();

            ExpectNear(0.5f, result.W(), 1e-4f);
        }

        // =========================================================================
        // Vector rotation
        // =========================================================================

        TEST_METHOD(RotateVectorByIdentity) {
            constexpr QuaternionSIMD<float> identity = QuaternionSIMD<float>::Identity();
            constexpr VecSIMD<float, 3> v{ 1.0f, 2.0f, 3.0f };

            const VecSIMD<float, 3> result = identity.RotateVector(v);

            ExpectNear(result, v);
        }

        TEST_METHOD(RotateVectorNinetyDegreesAroundZ) {
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);
            constexpr VecSIMD<float, 3> v{ 1.0f, 0.0f, 0.0f };

            const VecSIMD<float, 3> result = q.RotateVector(v);

            ExpectNear(result, VecSIMD<float, 3>{0.0f, 1.0f, 0.0f});
        }

        TEST_METHOD(RotateVectorOneEightyAroundX) {
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{1.0f, 0.0f, 0.0f}, kPi);
            constexpr VecSIMD<float, 3> v{ 0.0f, 1.0f, 0.0f };

            const VecSIMD<float, 3> result = q.RotateVector(v);

            ExpectNear(result, VecSIMD<float, 3>{0.0f, -1.0f, 0.0f});
        }

        TEST_METHOD(RotateVectorPreservesLength) {
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{1.0f, 1.0f, 1.0f}, kPi / 5.0f);
            constexpr VecSIMD<float, 3> v{ 3.0f, 4.0f, 0.0f };

            const VecSIMD<float, 3> result = q.RotateVector(v);

            const float originalLength = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            const float rotatedLength = std::sqrt(result[0] * result[0] + result[1] * result[1] + result[2] * result[2]);

            ExpectNear(rotatedLength, originalLength, 1e-4f);
        }

        TEST_METHOD(OperatorMultiplyRotatesVector) {
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);
            constexpr VecSIMD<float, 3> v{ 1.0f, 0.0f, 0.0f };

            const VecSIMD<float, 3> result = q * v;

            ExpectNear(result, q.RotateVector(v));
        }

        TEST_METHOD(OperatorMultiplyRotateVectorCommonType) {
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);
            constexpr VecSIMD<double, 3> v{ 1.0, 0.0, 0.0 };

            const auto result = q * v;

            static_assert(std::is_same_v<decltype(result), const VecSIMD<double, 3>>);

            ExpectNear(0.0, result[0], 1e-4);
            ExpectNear(1.0, result[1], 1e-4);
        }

        // =========================================================================
        // Euler conversions
        // =========================================================================

        TEST_METHOD(EulerRoundTripPitch) {
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromEuler(kPi / 6.0f, 0.0f, 0.0f);

            ExpectNear(kPi / 6.0f, q.Pitch(), 1e-4f);
            ExpectNear(0.0f, q.Yaw(), 1e-4f);
            ExpectNear(0.0f, q.Roll(), 1e-4f);
        }

        TEST_METHOD(EulerRoundTripYaw) {
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromEuler(0.0f, kPi / 6.0f, 0.0f);

            ExpectNear(0.0f, q.Pitch(), 1e-4f);
            ExpectNear(kPi / 6.0f, q.Yaw(), 1e-4f);
            ExpectNear(0.0f, q.Roll(), 1e-4f);
        }

        TEST_METHOD(EulerRoundTripRoll) {
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromEuler(0.0f, 0.0f, kPi / 6.0f);

            ExpectNear(0.0f, q.Pitch(), 1e-4f);
            ExpectNear(0.0f, q.Yaw(), 1e-4f);
            ExpectNear(kPi / 6.0f, q.Roll(), 1e-4f);
        }

        TEST_METHOD(ToEulerMatchesComponentAccessors) {
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromEuler(kPi / 8.0f, kPi / 5.0f, kPi / 7.0f);

            const VecSIMD<float, 3> euler = q.ToEuler();

            ExpectNear(q.Pitch(), euler[0], 1e-4f);
            ExpectNear(q.Yaw(), euler[1], 1e-4f);
            ExpectNear(q.Roll(), euler[2], 1e-4f);
        }

        TEST_METHOD(YawClampsAtGimbalLock) {
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromEuler(0.0f, kPi / 2.0f, 0.0f);

            ExpectNear(kPi / 2.0f, q.Yaw(), 1e-3f);
        }

        // =========================================================================
        // ToAxisAngle
        // =========================================================================

        TEST_METHOD(ToAxisAngleRoundTripsFromAxisAngle) {
            constexpr VecSIMD<float, 3> axis{ 0.0f, 1.0f, 0.0f };
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromAxisAngle(axis, kPi / 2.0f);

            VecSIMD<float, 3> outAxis;
            float outAngle{};
            q.ToAxisAngle(outAxis, outAngle);

            ExpectNear(outAxis, axis);
            ExpectNear(kPi / 2.0f, outAngle, 1e-4f);
        }

        TEST_METHOD(ToAxisAngleIdentityHasZeroAngle) {
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::Identity();

            VecSIMD<float, 3> outAxis;
            float outAngle{};
            q.ToAxisAngle(outAxis, outAngle);

            ExpectNear(0.0f, outAngle, 1e-4f);
        }

        TEST_METHOD(ToAxisAngleFullTurn) {
            constexpr VecSIMD<float, 3> axis{ 1.0f, 0.0f, 0.0f };
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromAxisAngle(axis, kPi);

            VecSIMD<float, 3> outAxis;
            float outAngle{};
            q.ToAxisAngle(outAxis, outAngle);

            ExpectNear(outAxis, axis);
            ExpectNear(kPi, outAngle, 1e-4f);
        }

        // =========================================================================
        // Dot / Lerp / Slerp
        // =========================================================================

        TEST_METHOD(DotOfIdenticalQuaternionSIMDsEqualsLengthSquared) {
            constexpr QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            ExpectNear(q.LengthSquared(), Dot(q, q));
        }

        TEST_METHOD(DotOfOrthogonalQuaternionSIMDsIsZero) {
            constexpr QuaternionSIMD<float> a{ 1.0f, 0.0f, 0.0f, 0.0f };
            constexpr QuaternionSIMD<float> b{ 0.0f, 1.0f, 0.0f, 0.0f };

            ExpectNear(0.0f, Dot(a, b));
        }

        TEST_METHOD(LerpAtStart) {
            const QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, 0.0f);
            const QuaternionSIMD<float> b = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);

            const QuaternionSIMD<float> result = Lerp(a, b, 0.0f);

            ExpectNear(result, a);
        }

        TEST_METHOD(LerpAtEnd) {
            const QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, 0.0f);
            const QuaternionSIMD<float> b = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);

            const QuaternionSIMD<float> result = Lerp(a, b, 1.0f);

            ExpectNear(result, b);
        }

        TEST_METHOD(LerpIsNormalized) {
            const QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, 0.0f);
            const QuaternionSIMD<float> b = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{1.0f, 0.0f, 0.0f}, kPi / 2.0f);

            const QuaternionSIMD<float> result = Lerp(a, b, 0.5f);

            ExpectNear(1.0f, result.Length(), 1e-4f);
        }

        TEST_METHOD(SlerpAtStart) {
            const QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, 0.0f);
            const QuaternionSIMD<float> b = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);

            const QuaternionSIMD<float> result = Slerp(a, b, 0.0f);

            ExpectNear(result, a);
        }

        TEST_METHOD(SlerpAtEnd) {
            const QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, 0.0f);
            const QuaternionSIMD<float> b = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);

            const QuaternionSIMD<float> result = Slerp(a, b, 1.0f);

            ExpectNear(result, b);
        }

        TEST_METHOD(SlerpAtMidpointMatchesHalfAngle) {
            const QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, 0.0f);
            const QuaternionSIMD<float> b = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);
            const QuaternionSIMD<float> expected =
                QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, kPi / 4.0f);

            const QuaternionSIMD<float> result = Slerp(a, b, 0.5f);

            ExpectNear(result, expected);
        }

        TEST_METHOD(SlerpIsNormalized) {
            const QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, 0.0f);
            const QuaternionSIMD<float> b = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{1.0f, 0.0f, 0.0f}, kPi / 2.0f);

            const QuaternionSIMD<float> result = Slerp(a, b, 0.3f);

            ExpectNear(1.0f, result.Length(), 1e-4f);
        }

        TEST_METHOD(SlerpTakesShortestPath) {
            const QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, 0.0f);
            QuaternionSIMD<float> b = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);
            b = -b; // Negate to represent the same rotation via the opposite hemisphere.

            const QuaternionSIMD<float> result = Slerp(a, b, 0.5f);
            const QuaternionSIMD<float> expected =
                QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 4.0f);

            ExpectNear(result, expected);
        }

        TEST_METHOD(SlerpNearIdenticalQuaternionSIMDsFallsBackToLerp) {
            const QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, 0.0f);
            const QuaternionSIMD<float> b = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, 1e-7f);

            const QuaternionSIMD<float> result = Slerp(a, b, 0.5f);

            ExpectNear(1.0f, result.Length(), 1e-4f);
        }

        // =========================================================================
        // LerpUnclamped / SlerpUnclamped
        // =========================================================================

        TEST_METHOD(LerpUnclampedMatchesLerpWithinRange) {
            const QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, 0.0f);
            const QuaternionSIMD<float> b = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);

            ExpectNear(LerpUnclamped(a, b, 0.5f), Lerp(a, b, 0.5f));
        }

        TEST_METHOD(LerpUnclampedExtrapolatesPastEnd) {
            const QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, 0.0f);
            const QuaternionSIMD<float> b = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);

            const QuaternionSIMD<float> clamped = Lerp(a, b, 1.5f);
            const QuaternionSIMD<float> unclamped = LerpUnclamped(a, b, 1.5f);

            ExpectNear(1.0f, unclamped.Length(), 1e-4f);
            Assert::IsTrue(std::abs(unclamped.Z() - clamped.Z()) > 1e-4f);
        }

        TEST_METHOD(SlerpUnclampedMatchesSlerpWithinRange) {
            const QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, 0.0f);
            const QuaternionSIMD<float> b = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);

            ExpectNear(SlerpUnclamped(a, b, 0.5f), Slerp(a, b, 0.5f));
        }

        TEST_METHOD(SlerpUnclampedIsNormalizedPastEnd) {
            const QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, 0.0f);
            const QuaternionSIMD<float> b = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);

            const QuaternionSIMD<float> result = SlerpUnclamped(a, b, 1.5f);

            ExpectNear(1.0f, result.Length(), 1e-4f);
        }

        // =========================================================================
        // Angle
        // =========================================================================

        TEST_METHOD(AngleBetweenIdenticalQuaternionSIMDsIsZero) {
            const QuaternionSIMD<float> q = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, kPi / 3.0f);

            ExpectNear(0.0f, Angle(q, q), 1e-4f);
        }

        TEST_METHOD(AngleBetweenIdentityAndRotatedQuaternionSIMD) {
            const QuaternionSIMD<float> identity = QuaternionSIMD<float>::Identity();
            const QuaternionSIMD<float> rotated = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);

            ExpectNear(kPi / 2.0f, Angle(identity, rotated), 1e-4f);
        }

        TEST_METHOD(AngleIsSignInvariant) {
            const QuaternionSIMD<float> a = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);
            const QuaternionSIMD<float> b = -a;

            // acos loses precision near its domain boundary, so this needs a looser tolerance than usual.
            ExpectNear(0.0f, Angle(a, b), 1e-2f);
        }

        // =========================================================================
        // RotateTowards
        // =========================================================================

        TEST_METHOD(RotateTowardsPartialStep) {
            const QuaternionSIMD<float> from = QuaternionSIMD<float>::Identity();
            const QuaternionSIMD<float> to = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);

            const QuaternionSIMD<float> result = RotateTowards(from, to, kPi / 4.0f);

            ExpectNear(kPi / 4.0f, Angle(from, result), 1e-3f);
        }

        TEST_METHOD(RotateTowardsOvershootClampsToTarget) {
            const QuaternionSIMD<float> from = QuaternionSIMD<float>::Identity();
            const QuaternionSIMD<float> to = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);

            const QuaternionSIMD<float> result = RotateTowards(from, to, kPi);

            ExpectNear(result, to);
        }

        TEST_METHOD(RotateTowardsSameRotationReturnsTarget) {
            const QuaternionSIMD<float> from = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 3.0f);
            const QuaternionSIMD<float> to = QuaternionSIMD<float>::FromAxisAngle(VecSIMD<float, 3>{0.0f, 0.0f, 1.0f}, kPi / 3.0f);

            const QuaternionSIMD<float> result = RotateTowards(from, to, kPi / 4.0f);

            ExpectNear(result, to);
        }

        // =========================================================================
        // Double precision / 256-bit SIMD
        // =========================================================================

        TEST_METHOD(DoubleAddition) {
            constexpr QuaternionSIMD<double> a{ 1.0, 2.0, 3.0, 4.0 };
            constexpr QuaternionSIMD<double> b{ 4.0, 5.0, 6.0, 7.0 };
            const auto result = a + b;

            ExpectNear(5.0, result.X(), 1e-9);
            ExpectNear(7.0, result.Y(), 1e-9);
            ExpectNear(9.0, result.Z(), 1e-9);
            ExpectNear(11.0, result.W(), 1e-9);
        }

        TEST_METHOD(DoubleSubtraction) {
            constexpr QuaternionSIMD<double> a{ 5.0, 7.0, 9.0, 11.0 };
            constexpr QuaternionSIMD<double> b{ 1.0, 2.0, 3.0, 4.0 };
            const auto result = a - b;

            ExpectNear(4.0, result.X(), 1e-9);
            ExpectNear(5.0, result.Y(), 1e-9);
            ExpectNear(6.0, result.Z(), 1e-9);
            ExpectNear(7.0, result.W(), 1e-9);
        }

        TEST_METHOD(DoubleMultiplication) {
            constexpr QuaternionSIMD<double> a{ 0.0, 0.0, 0.0, 1.0 };
            constexpr QuaternionSIMD<double> b{ 1.0, 0.0, 0.0, 0.0 };
            const auto result = a * b;

            ExpectNear(1.0, result.X(), 1e-9);
            ExpectNear(0.0, result.Y(), 1e-9);
            ExpectNear(0.0, result.Z(), 1e-9);
            ExpectNear(0.0, result.W(), 1e-9);
        }

        TEST_METHOD(DoubleScalarMultiplication) {
            constexpr QuaternionSIMD<double> q{ 1.0, 2.0, 3.0, 4.0 };
            const auto result = q * 2.0;

            ExpectNear(2.0, result.X(), 1e-9);
            ExpectNear(4.0, result.Y(), 1e-9);
            ExpectNear(6.0, result.Z(), 1e-9);
            ExpectNear(8.0, result.W(), 1e-9);
        }

        TEST_METHOD(DoubleScalarDivision) {
            constexpr QuaternionSIMD<double> q{ 2.0, 4.0, 6.0, 8.0 };
            const auto result = q / 2.0;

            ExpectNear(1.0, result.X(), 1e-9);
            ExpectNear(2.0, result.Y(), 1e-9);
            ExpectNear(3.0, result.Z(), 1e-9);
            ExpectNear(4.0, result.W(), 1e-9);
        }

        TEST_METHOD(DoubleLengthSquared) {
            constexpr QuaternionSIMD<double> q{ 1.0, 2.0, 2.0, 4.0 };
            ExpectNear(25.0, q.LengthSquared(), 1e-9);
        }

        TEST_METHOD(DoubleLength) {
            constexpr QuaternionSIMD<double> q{ 0.0, 0.0, 0.0, 3.0 };
            ExpectNear(3.0, q.Length(), 1e-9);
        }

        TEST_METHOD(DoubleNormalized) {
            constexpr QuaternionSIMD<double> q{ 0.0, 0.0, 0.0, 2.0 };
            const QuaternionSIMD<double> result = q.Normalized();

            ExpectNear(1.0, result.Length(), 1e-9);
            ExpectNear(1.0, result.W(), 1e-9);
        }

        TEST_METHOD(DoubleConjugate) {
            constexpr QuaternionSIMD<double> q{ 1.0, 2.0, 3.0, 4.0 };
            const auto result = q.Conjugate();

            ExpectNear(-1.0, result.X(), 1e-9);
            ExpectNear(-2.0, result.Y(), 1e-9);
            ExpectNear(-3.0, result.Z(), 1e-9);
            ExpectNear(4.0, result.W(), 1e-9);
        }

        TEST_METHOD(DoubleDot) {
            constexpr QuaternionSIMD<double> q{ 1.0, 2.0, 3.0, 4.0 };
            ExpectNear(30.0, Dot(q, q), 1e-9);
        }

        TEST_METHOD(DoubleEquality) {
            constexpr QuaternionSIMD<double> a{ 1.0, 2.0, 3.0, 4.0 };
            constexpr QuaternionSIMD<double> b{ 1.0, 2.0, 3.0, 4.0 };
            Assert::IsTrue(a == b);
        }

        TEST_METHOD(DoubleInequality) {
            constexpr QuaternionSIMD<double> a{ 1.0, 2.0, 3.0, 4.0 };
            constexpr QuaternionSIMD<double> b{ 1.0, 2.0, 3.0, 5.0 };
            Assert::IsFalse(a == b);
        }

        TEST_METHOD(DoubleFromAxisAngle) {
            constexpr VecSIMD<double, 3> axis{ 0.0, 1.0, 0.0 };
            const QuaternionSIMD<double> q =
                QuaternionSIMD<double>::FromAxisAngle(axis, std::numbers::pi_v<double> / 2.0);

            ExpectNear(0.0, q.X(), 1e-9);
            ExpectNear(std::sin(std::numbers::pi_v<double> / 4.0), q.Y(), 1e-9);
            ExpectNear(0.0, q.Z(), 1e-9);
            ExpectNear(std::cos(std::numbers::pi_v<double> / 4.0), q.W(), 1e-9);
        }

        TEST_METHOD(DoubleRotation) {
            constexpr VecSIMD<double, 3> axis{ 0.0, 0.0, 1.0 };
            constexpr VecSIMD<double, 3> vector{ 1.0, 0.0, 0.0 };
            const QuaternionSIMD<double> q =
                QuaternionSIMD<double>::FromAxisAngle(axis, std::numbers::pi_v<double> / 2.0);

            const VecSIMD<double, 3> result = q.RotateVector(vector);

            ExpectNear(0.0, result[0], 1e-9);
            ExpectNear(1.0, result[1], 1e-9);
            ExpectNear(0.0, result[2], 1e-9);
        }

        TEST_METHOD(DoubleMultiplicationByIdentity) {
            constexpr QuaternionSIMD<double> q{ 1.0, 2.0, 3.0, 4.0 };
            constexpr QuaternionSIMD<double> identity = QuaternionSIMD<double>::Identity();
            const auto result = q * identity;

            ExpectNear(q, result, 1e-9);
        }

        TEST_METHOD(DoubleInverseTimesSelfIsIdentity) {
            const QuaternionSIMD<double> q = QuaternionSIMD<double>::FromAxisAngle(
                VecSIMD<double, 3>{ 1.0, 1.0, 0.0 }, std::numbers::pi_v<double> / 4.0);
            const QuaternionSIMD<double> result = q * q.Inverse();

            ExpectNear(result, QuaternionSIMD<double>::Identity(), 1e-9);
        }

        // =========================================================================
        // Constexpr
        // =========================================================================

        TEST_METHOD(ConstexprConstruction) {
            constexpr QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            static_assert(q.X() == 1.0f);
            static_assert(q.Y() == 2.0f);
            static_assert(q.Z() == 3.0f);
            static_assert(q.W() == 4.0f);
        }

        TEST_METHOD(ConstexprArithmetic) {
            constexpr QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr QuaternionSIMD<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

            const auto sum = a + b;
            const auto difference = a - b;

            Assert::AreEqual(sum.X(), 5.0f);
            Assert::AreEqual(sum.Y(), 7.0f);
            Assert::AreEqual(sum.Z(), 9.0f);
            Assert::AreEqual(sum.W(), 11.0f);

            Assert::AreEqual(difference.X(), -3.0f);
            Assert::AreEqual(difference.Y(), -3.0f);
            Assert::AreEqual(difference.Z(), -3.0f);
            Assert::AreEqual(difference.W(), -3.0f);
        }

        // =========================================================================
        // Formatting
        // =========================================================================

        TEST_METHOD(Formatting) {
            constexpr QuaternionSIMD<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            Assert::AreEqual(std::string("(1, 2, 3, 4)"), std::format("{}", q));
        }

        TEST_METHOD(FormattingFloatingPoint) {
            constexpr QuaternionSIMD<float> q{ 1.0f, 2.5f, 3.25f, 4.0f };

            Assert::AreEqual(std::string("(1, 2.5, 3.25, 4)"), std::format("{}", q));
        }
    };
}
