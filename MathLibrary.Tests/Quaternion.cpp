#include "pch.h"
#include "CppUnitTest.h"

#include "Headers/Quaternion.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace {
    constexpr float kPi = std::numbers::pi_v<float>;

    // Compares two floating-point values through Assert::AreEqual's tolerance overload.
    void ExpectNear(double expected, double actual, double tolerance = 1e-4) {
        Assert::AreEqual(expected, actual, tolerance);
    }

    template <float_num T>
    void ExpectNear(const Quaternion<T>& a, const Quaternion<T>& b, T tolerance = static_cast<T>(1e-4)) {
        ExpectNear(a.X(), b.X(), tolerance);
        ExpectNear(a.Y(), b.Y(), tolerance);
        ExpectNear(a.Z(), b.Z(), tolerance);
        ExpectNear(a.W(), b.W(), tolerance);
    }

    template <float_num T>
    void ExpectNear(const Vec3<T>& a, const Vec3<T>& b, T tolerance = static_cast<T>(1e-4)) {
        ExpectNear(a[0], b[0], tolerance);
        ExpectNear(a[1], b[1], tolerance);
        ExpectNear(a[2], b[2], tolerance);
    }
} // namespace

namespace MathLibraryTests
{
    TEST_CLASS(QuaternionTests)
    {
    public:
        // =========================================================================
        // Construction
        // =========================================================================

        TEST_METHOD(DefaultConstruction) {
            constexpr Quaternion<float> q{};

            ExpectNear(0.0f, q.X());
            ExpectNear(0.0f, q.Y());
            ExpectNear(0.0f, q.Z());
            ExpectNear(1.0f, q.W());
        }

        TEST_METHOD(ValueConstruction) {
            constexpr Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            ExpectNear(1.0f, q.X());
            ExpectNear(2.0f, q.Y());
            ExpectNear(3.0f, q.Z());
            ExpectNear(4.0f, q.W());
        }

        TEST_METHOD(Identity) {
            constexpr Quaternion<float> q = Quaternion<float>::Identity();

            ExpectNear(0.0f, q.X());
            ExpectNear(0.0f, q.Y());
            ExpectNear(0.0f, q.Z());
            ExpectNear(1.0f, q.W());
        }

        TEST_METHOD(FromAxisAngle) {
            constexpr Vec3<float> axis{ 0.0f, 1.0f, 0.0f };
            const Quaternion<float> q = Quaternion<float>::FromAxisAngle(axis, kPi / 2.0f);

            ExpectNear(0.0f, q.X(), 1e-4f);
            ExpectNear(std::sin(kPi / 4.0f), q.Y(), 1e-4f);
            ExpectNear(0.0f, q.Z(), 1e-4f);
            ExpectNear(std::cos(kPi / 4.0f), q.W(), 1e-4f);
        }

        TEST_METHOD(FromAxisAngleNormalizesAxis) {
            constexpr Vec3<float> axis{ 0.0f, 5.0f, 0.0f };
            const Quaternion<float> q = Quaternion<float>::FromAxisAngle(axis, kPi / 2.0f);

            ExpectNear(1.0f, q.Length(), 1e-4f);
            ExpectNear(std::sin(kPi / 4.0f), q.Y(), 1e-4f);
        }

        TEST_METHOD(FromAxisAngleZeroAxis) {
            constexpr Vec3<float> axis{ 0.0f, 0.0f, 0.0f };
            const Quaternion<float> q = Quaternion<float>::FromAxisAngle(axis, kPi / 2.0f);

            ExpectNear(0.0f, q.X(), 1e-4f);
            ExpectNear(0.0f, q.Y(), 1e-4f);
            ExpectNear(0.0f, q.Z(), 1e-4f);
        }

        TEST_METHOD(AxisAngleConstructor) {
            constexpr Vec3<float> axis{ 0.0f, 0.0f, 1.0f };
            const Quaternion<float> q{ axis, kPi };

            ExpectNear(q, Quaternion<float>::FromAxisAngle(axis, kPi));
        }

        TEST_METHOD(FromEulerIdentity) {
            const Quaternion<float> q = Quaternion<float>::FromEuler(0.0f, 0.0f, 0.0f);

            ExpectNear(q, Quaternion<float>::Identity());
        }

        TEST_METHOD(FromEulerMatchesAxisAngleForSingleAxis) {
            constexpr float angle = kPi / 3.0f;

            const Quaternion<float> fromEuler = Quaternion<float>::FromEuler(angle, 0.0f, 0.0f);
            const Quaternion<float> fromAxisAngle =
                Quaternion<float>::FromAxisAngle(Vec3<float>{1.0f, 0.0f, 0.0f}, angle);

            ExpectNear(fromEuler, fromAxisAngle);
        }

        TEST_METHOD(MixedValueTypes) {
            constexpr Quaternion<double> q{ 1, 2.5, 3, 4 };

            ExpectNear(1.0, q.X(), 1e-9);
            ExpectNear(2.5, q.Y(), 1e-9);
            ExpectNear(3.0, q.Z(), 1e-9);
            ExpectNear(4.0, q.W(), 1e-9);
        }

        // =========================================================================
        // FromToRotation
        // =========================================================================

        TEST_METHOD(FromToRotationIdenticalVectors) {
            constexpr Vec3<float> v{ 1.0f, 0.0f, 0.0f };

            const Quaternion<float> q = Quaternion<float>::FromToRotation(v, v);

            ExpectNear(q, Quaternion<float>::Identity());
        }

        TEST_METHOD(FromToRotationOpposingVectors) {
            constexpr Vec3<float> from{ 1.0f, 0.0f, 0.0f };
            constexpr Vec3<float> to{ -1.0f, 0.0f, 0.0f };

            const Quaternion<float> q = Quaternion<float>::FromToRotation(from, to);

            ExpectNear(1.0f, q.Length(), 1e-4f);

            const Vec3<float> rotated = q.RotateVector(from);

            ExpectNear(rotated, to);
        }

        TEST_METHOD(FromToRotationRotatesFromOntoTo) {
            constexpr Vec3<float> from{ 1.0f, 0.0f, 0.0f };
            constexpr Vec3<float> to{ 0.0f, 1.0f, 0.0f };

            const Quaternion<float> q = Quaternion<float>::FromToRotation(from, to);
            const Vec3<float> rotated = q.RotateVector(from);

            ExpectNear(rotated, to);
        }

        TEST_METHOD(FromToRotationIsNormalized) {
            constexpr Vec3<float> from{ 1.0f, 0.0f, 0.0f };
            constexpr Vec3<float> to{ 0.0f, 0.0f, 1.0f };

            const Quaternion<float> q = Quaternion<float>::FromToRotation(from, to);

            ExpectNear(1.0f, q.Length(), 1e-4f);
        }

        // =========================================================================
        // LookRotation
        // =========================================================================

        TEST_METHOD(LookRotationIdentityForForwardZ) {
            constexpr Vec3<float> forward{ 0.0f, 0.0f, 1.0f };
            constexpr Vec3<float> up{ 0.0f, 1.0f, 0.0f };

            const Quaternion<float> q = Quaternion<float>::LookRotation(forward, up);

            ExpectNear(q, Quaternion<float>::Identity());
        }

        TEST_METHOD(LookRotationRotatesForwardVector) {
            constexpr Vec3<float> forward{ 1.0f, 0.0f, 0.0f };
            constexpr Vec3<float> up{ 0.0f, 1.0f, 0.0f };

            const Quaternion<float> q = Quaternion<float>::LookRotation(forward, up);
            const Vec3<float> rotated = q.RotateVector(Vec3<float>{0.0f, 0.0f, 1.0f});

            ExpectNear(rotated, forward);
        }

        TEST_METHOD(LookRotationIsNormalized) {
            constexpr Vec3<float> forward{ 0.0f, 0.0f, 1.0f };
            constexpr Vec3<float> up{ 0.0f, 1.0f, 0.0f };

            const Quaternion<float> q = Quaternion<float>::LookRotation(forward, up);

            ExpectNear(1.0f, q.Length(), 1e-4f);
        }

        // =========================================================================
        // Access
        // =========================================================================

        TEST_METHOD(ComponentAccess) {
            Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

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
            constexpr Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            ExpectNear(1.0f, q.X());
            ExpectNear(2.0f, q.Y());
            ExpectNear(3.0f, q.Z());
            ExpectNear(4.0f, q.W());
        }

        TEST_METHOD(Data) {
            Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

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
            Assert::AreEqual(static_cast<std::size_t>(4), static_cast<std::size_t>((Quaternion<float>::Size())));
            Assert::AreEqual(static_cast<std::size_t>(4), static_cast<std::size_t>((Quaternion<double>::Size())));
        }

        // =========================================================================
        // Equality
        // =========================================================================

        TEST_METHOD(Equality) {
            constexpr Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr Quaternion<float> b{ 1.0f, 2.0f, 3.0f, 4.0f };

            Assert::IsTrue(a == b);
        }

        TEST_METHOD(Inequality) {
            constexpr Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr Quaternion<float> b{ 1.0f, 2.0f, 3.0f, 5.0f };

            Assert::IsFalse(a == b);
        }

        TEST_METHOD(EqualityDifferentTypes) {
            constexpr Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr Quaternion<double> b{ 1.0, 2.0, 3.0, 4.0 };

            Assert::IsTrue(a == b);
        }

        // =========================================================================
        // Addition
        // =========================================================================

        TEST_METHOD(Addition) {
            constexpr Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr Quaternion<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

            constexpr auto result = a + b;

            ExpectNear(5.0f, result.X());
            ExpectNear(7.0f, result.Y());
            ExpectNear(9.0f, result.Z());
            ExpectNear(11.0f, result.W());
        }

        TEST_METHOD(AdditionCommonType) {
            constexpr Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr Quaternion<double> b{ 0.5, 1.5, 2.5, 3.5 };

            constexpr auto result = a + b;

            static_assert(std::is_same_v<decltype(result), const Quaternion<double>>);

            ExpectNear(1.5, result.X(), 1e-9);
            ExpectNear(3.5, result.Y(), 1e-9);
            ExpectNear(5.5, result.Z(), 1e-9);
            ExpectNear(7.5, result.W(), 1e-9);
        }

        TEST_METHOD(AdditionAssignment) {
            Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr Quaternion<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

            a += b;

            ExpectNear(5.0f, a.X());
            ExpectNear(7.0f, a.Y());
            ExpectNear(9.0f, a.Z());
            ExpectNear(11.0f, a.W());
        }

        TEST_METHOD(MixedTypeAdditionAssignment) {
            Quaternion<double> a{ 1.0, 2.0, 3.0, 4.0 };
            constexpr Quaternion<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

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
            constexpr Quaternion<float> a{ 5.0f, 7.0f, 9.0f, 11.0f };
            constexpr Quaternion<float> b{ 1.0f, 2.0f, 3.0f, 4.0f };

            constexpr auto result = a - b;

            ExpectNear(4.0f, result.X());
            ExpectNear(5.0f, result.Y());
            ExpectNear(6.0f, result.Z());
            ExpectNear(7.0f, result.W());
        }

        TEST_METHOD(SubtractionCommonType) {
            constexpr Quaternion<float> a{ 5.0f, 7.0f, 9.0f, 11.0f };
            constexpr Quaternion<double> b{ 0.5, 1.5, 2.5, 3.5 };

            constexpr auto result = a - b;

            static_assert(std::is_same_v<decltype(result), const Quaternion<double>>);

            ExpectNear(4.5, result.X(), 1e-9);
            ExpectNear(5.5, result.Y(), 1e-9);
            ExpectNear(6.5, result.Z(), 1e-9);
            ExpectNear(7.5, result.W(), 1e-9);
        }

        TEST_METHOD(SubtractionAssignment) {
            Quaternion<float> a{ 5.0f, 7.0f, 9.0f, 11.0f };
            constexpr Quaternion<float> b{ 1.0f, 2.0f, 3.0f, 4.0f };

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
            constexpr Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr Quaternion<float> identity = Quaternion<float>::Identity();

            constexpr auto result = q * identity;

            ExpectNear(q.X(), result.X());
            ExpectNear(q.Y(), result.Y());
            ExpectNear(q.Z(), result.Z());
            ExpectNear(q.W(), result.W());
        }

        TEST_METHOD(MultiplicationIsNonCommutative) {
            const Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{1.0f, 0.0f, 0.0f}, kPi / 2.0f);
            const Quaternion<float> b = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);

            const Quaternion<float> ab = a * b;
            const Quaternion<float> ba = b * a;

            Assert::IsFalse((ab.X() == ba.X()) && (ab.Y() == ba.Y()) && (ab.Z() == ba.Z()) && (ab.W() == ba.W()));
        }

        TEST_METHOD(MultiplicationCombinesRotations) {
            // Two 90 degree rotations around Z should equal one 180 degree rotation around Z.
            const Quaternion<float> quarterTurn =
                Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);
            const Quaternion<float> halfTurn = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi);

            const Quaternion<float> combined = quarterTurn * quarterTurn;

            ExpectNear(combined, halfTurn);
        }

        TEST_METHOD(MultiplicationCommonType) {
            constexpr Quaternion<float> a{ 0.0f, 0.0f, 0.0f, 1.0f };
            constexpr Quaternion<double> b{ 1.0, 0.0, 0.0, 0.0 };

            constexpr auto result = a * b;

            static_assert(std::is_same_v<decltype(result), const Quaternion<double>>);

            ExpectNear(1.0, result.X(), 1e-9);
            ExpectNear(0.0, result.Y(), 1e-9);
            ExpectNear(0.0, result.Z(), 1e-9);
            ExpectNear(0.0, result.W(), 1e-9);
        }

        TEST_METHOD(MultiplicationAssignment) {
            Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);
            const Quaternion<float> b = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);

            a *= b;

            ExpectNear(a, Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi));
        }

        // =========================================================================
        // Scalar multiplication
        // =========================================================================

        TEST_METHOD(ScalarMultiplication) {
            constexpr Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            constexpr auto result = q * 2.0f;

            ExpectNear(2.0f, result.X());
            ExpectNear(4.0f, result.Y());
            ExpectNear(6.0f, result.Z());
            ExpectNear(8.0f, result.W());
        }

        TEST_METHOD(ScalarMultiplicationCommonType) {
            constexpr Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            constexpr auto result = q * 0.5;

            static_assert(std::is_same_v<decltype(result), const Quaternion<double>>);

            ExpectNear(0.5, result.X(), 1e-9);
            ExpectNear(1.0, result.Y(), 1e-9);
            ExpectNear(1.5, result.Z(), 1e-9);
            ExpectNear(2.0, result.W(), 1e-9);
        }

        TEST_METHOD(ScalarMultiplicationReversed) {
            constexpr Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            constexpr auto result = 2.0f * q;

            ExpectNear(2.0f, result.X());
            ExpectNear(4.0f, result.Y());
            ExpectNear(6.0f, result.Z());
            ExpectNear(8.0f, result.W());
        }

        TEST_METHOD(ScalarMultiplicationAssignment) {
            Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

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
            constexpr Quaternion<float> q{ 2.0f, 4.0f, 6.0f, 8.0f };

            constexpr auto result = q / 2.0f;

            ExpectNear(1.0f, result.X());
            ExpectNear(2.0f, result.Y());
            ExpectNear(3.0f, result.Z());
            ExpectNear(4.0f, result.W());
        }

        TEST_METHOD(ScalarDivisionCommonType) {
            constexpr Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            constexpr auto result = q / 2.0;

            static_assert(std::is_same_v<decltype(result), const Quaternion<double>>);

            ExpectNear(0.5, result.X(), 1e-9);
            ExpectNear(1.0, result.Y(), 1e-9);
            ExpectNear(1.5, result.Z(), 1e-9);
            ExpectNear(2.0, result.W(), 1e-9);
        }

        TEST_METHOD(ScalarDivisionAssignment) {
            Quaternion<float> q{ 2.0f, 4.0f, 6.0f, 8.0f };

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
            constexpr Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            constexpr auto result = +q;

            ExpectNear(1.0f, result.X());
            ExpectNear(2.0f, result.Y());
            ExpectNear(3.0f, result.Z());
            ExpectNear(4.0f, result.W());
        }

        TEST_METHOD(UnaryMinus) {
            constexpr Quaternion<float> q{ 1.0f, -2.0f, 3.0f, -4.0f };

            constexpr auto result = -q;

            ExpectNear(-1.0f, result.X());
            ExpectNear(2.0f, result.Y());
            ExpectNear(-3.0f, result.Z());
            ExpectNear(4.0f, result.W());
        }

        // =========================================================================
        // Length / normalization
        // =========================================================================

        TEST_METHOD(LengthSquared) {
            constexpr Quaternion<float> q{ 1.0f, 2.0f, 2.0f, 4.0f };

            ExpectNear(25.0f, q.LengthSquared());
        }

        TEST_METHOD(Length) {
            constexpr Quaternion<float> q{ 0.0f, 0.0f, 0.0f, 3.0f };

            ExpectNear(3.0f, q.Length());
        }

        TEST_METHOD(IdentityLengthIsOne) {
            constexpr Quaternion<float> q = Quaternion<float>::Identity();

            ExpectNear(1.0f, q.Length());
        }

        TEST_METHOD(Normalized) {
            constexpr Quaternion<float> q{ 0.0f, 0.0f, 0.0f, 2.0f };

            const Quaternion<float> result = q.Normalized();

            ExpectNear(1.0f, result.Length());
            ExpectNear(1.0f, result.W());
        }

        TEST_METHOD(NormalizedDoesNotMutate) {
            constexpr Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            const Quaternion<float> result = q.Normalized();

            ExpectNear(1.0f, q.X());
            ExpectNear(2.0f, q.Y());
            ExpectNear(3.0f, q.Z());
            ExpectNear(4.0f, q.W());
            ExpectNear(1.0f, result.Length(), 1e-4f);
        }

        TEST_METHOD(NormalizedZeroQuaternion) {
            constexpr Quaternion<float> q{ 0.0f, 0.0f, 0.0f, 0.0f };

            const Quaternion<float> result = q.Normalized();

            ExpectNear(0.0f, result.X());
            ExpectNear(0.0f, result.Y());
            ExpectNear(0.0f, result.Z());
            ExpectNear(0.0f, result.W());
        }

        TEST_METHOD(Normalize) {
            Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            q.Normalize();

            ExpectNear(1.0f, q.Length(), 1e-4f);
        }

        // =========================================================================
        // Conjugate / inverse
        // =========================================================================

        TEST_METHOD(Conjugate) {
            constexpr Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            constexpr auto result = q.Conjugate();

            ExpectNear(-1.0f, result.X());
            ExpectNear(-2.0f, result.Y());
            ExpectNear(-3.0f, result.Z());
            ExpectNear(4.0f, result.W());
        }

        TEST_METHOD(InverseOfUnitQuaternionEqualsConjugate) {
            const Quaternion<float> q = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, kPi / 3.0f);

            ExpectNear(q.Inverse(), q.Conjugate());
        }

        TEST_METHOD(InverseTimesSelfIsIdentity) {
            const Quaternion<float> q = Quaternion<float>::FromAxisAngle(Vec3<float>{1.0f, 1.0f, 0.0f}, kPi / 4.0f);

            const Quaternion<float> result = q * q.Inverse();

            ExpectNear(result, Quaternion<float>::Identity());
        }

        TEST_METHOD(InverseOfZeroQuaternionReturnsConjugate) {
            constexpr Quaternion<float> q{ 0.0f, 0.0f, 0.0f, 0.0f };

            constexpr auto result = q.Inverse();

            ExpectNear(0.0f, result.X());
            ExpectNear(0.0f, result.Y());
            ExpectNear(0.0f, result.Z());
            ExpectNear(0.0f, result.W());
        }

        TEST_METHOD(InverseOfNonUnitQuaternion) {
            constexpr Quaternion<float> q{ 0.0f, 0.0f, 0.0f, 2.0f };

            constexpr Quaternion<float> result = q.Inverse();

            ExpectNear(0.5f, result.W(), 1e-4f);
        }

        // =========================================================================
        // Vector rotation
        // =========================================================================

        TEST_METHOD(RotateVectorByIdentity) {
            constexpr Quaternion<float> identity = Quaternion<float>::Identity();
            constexpr Vec3<float> v{ 1.0f, 2.0f, 3.0f };

            constexpr Vec3<float> result = identity.RotateVector(v);

            ExpectNear(result, v);
        }

        TEST_METHOD(RotateVectorNinetyDegreesAroundZ) {
            const Quaternion<float> q = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);
            constexpr Vec3<float> v{ 1.0f, 0.0f, 0.0f };

            const Vec3<float> result = q.RotateVector(v);

            ExpectNear(result, Vec3<float>{0.0f, 1.0f, 0.0f});
        }

        TEST_METHOD(RotateVectorOneEightyAroundX) {
            const Quaternion<float> q = Quaternion<float>::FromAxisAngle(Vec3<float>{1.0f, 0.0f, 0.0f}, kPi);
            constexpr Vec3<float> v{ 0.0f, 1.0f, 0.0f };

            const Vec3<float> result = q.RotateVector(v);

            ExpectNear(result, Vec3<float>{0.0f, -1.0f, 0.0f});
        }

        TEST_METHOD(RotateVectorPreservesLength) {
            const Quaternion<float> q = Quaternion<float>::FromAxisAngle(Vec3<float>{1.0f, 1.0f, 1.0f}, kPi / 5.0f);
            constexpr Vec3<float> v{ 3.0f, 4.0f, 0.0f };

            const Vec3<float> result = q.RotateVector(v);

            const float originalLength = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            const float rotatedLength = std::sqrt(result[0] * result[0] + result[1] * result[1] + result[2] * result[2]);

            ExpectNear(rotatedLength, originalLength, 1e-4f);
        }

        TEST_METHOD(OperatorMultiplyRotatesVector) {
            const Quaternion<float> q = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);
            constexpr Vec3<float> v{ 1.0f, 0.0f, 0.0f };

            const Vec3<float> result = q * v;

            ExpectNear(result, q.RotateVector(v));
        }

        TEST_METHOD(OperatorMultiplyRotateVectorCommonType) {
            const Quaternion<float> q = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);
            constexpr Vec3<double> v{ 1.0, 0.0, 0.0 };

            const auto result = q * v;

            static_assert(std::is_same_v<decltype(result), const Vec3<double>>);

            ExpectNear(0.0, result[0], 1e-4);
            ExpectNear(1.0, result[1], 1e-4);
        }

        // =========================================================================
        // Euler conversions
        // =========================================================================

        TEST_METHOD(EulerRoundTripPitch) {
            const Quaternion<float> q = Quaternion<float>::FromEuler(kPi / 6.0f, 0.0f, 0.0f);

            ExpectNear(kPi / 6.0f, q.Pitch(), 1e-4f);
            ExpectNear(0.0f, q.Yaw(), 1e-4f);
            ExpectNear(0.0f, q.Roll(), 1e-4f);
        }

        TEST_METHOD(EulerRoundTripYaw) {
            const Quaternion<float> q = Quaternion<float>::FromEuler(0.0f, kPi / 6.0f, 0.0f);

            ExpectNear(0.0f, q.Pitch(), 1e-4f);
            ExpectNear(kPi / 6.0f, q.Yaw(), 1e-4f);
            ExpectNear(0.0f, q.Roll(), 1e-4f);
        }

        TEST_METHOD(EulerRoundTripRoll) {
            const Quaternion<float> q = Quaternion<float>::FromEuler(0.0f, 0.0f, kPi / 6.0f);

            ExpectNear(0.0f, q.Pitch(), 1e-4f);
            ExpectNear(0.0f, q.Yaw(), 1e-4f);
            ExpectNear(kPi / 6.0f, q.Roll(), 1e-4f);
        }

        TEST_METHOD(ToEulerMatchesComponentAccessors) {
            const Quaternion<float> q = Quaternion<float>::FromEuler(kPi / 8.0f, kPi / 5.0f, kPi / 7.0f);

            const Vec3<float> euler = q.ToEuler();

            ExpectNear(q.Pitch(), euler[0], 1e-4f);
            ExpectNear(q.Yaw(), euler[1], 1e-4f);
            ExpectNear(q.Roll(), euler[2], 1e-4f);
        }

        TEST_METHOD(YawClampsAtGimbalLock) {
            const Quaternion<float> q = Quaternion<float>::FromEuler(0.0f, kPi / 2.0f, 0.0f);

            ExpectNear(kPi / 2.0f, q.Yaw(), 1e-3f);
        }

        // =========================================================================
        // ToAxisAngle
        // =========================================================================

        TEST_METHOD(ToAxisAngleRoundTripsFromAxisAngle) {
            constexpr Vec3<float> axis{ 0.0f, 1.0f, 0.0f };
            const Quaternion<float> q = Quaternion<float>::FromAxisAngle(axis, kPi / 2.0f);

            Vec3<float> outAxis;
            float outAngle{};
            q.ToAxisAngle(outAxis, outAngle);

            ExpectNear(outAxis, axis);
            ExpectNear(kPi / 2.0f, outAngle, 1e-4f);
        }

        TEST_METHOD(ToAxisAngleIdentityHasZeroAngle) {
            const Quaternion<float> q = Quaternion<float>::Identity();

            Vec3<float> outAxis;
            float outAngle{};
            q.ToAxisAngle(outAxis, outAngle);

            ExpectNear(0.0f, outAngle, 1e-4f);
        }

        TEST_METHOD(ToAxisAngleFullTurn) {
            constexpr Vec3<float> axis{ 1.0f, 0.0f, 0.0f };
            const Quaternion<float> q = Quaternion<float>::FromAxisAngle(axis, kPi);

            Vec3<float> outAxis;
            float outAngle{};
            q.ToAxisAngle(outAxis, outAngle);

            ExpectNear(outAxis, axis);
            ExpectNear(kPi, outAngle, 1e-4f);
        }

        // =========================================================================
        // Dot / Lerp / Slerp
        // =========================================================================

        TEST_METHOD(DotOfIdenticalQuaternionsEqualsLengthSquared) {
            constexpr Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            ExpectNear(q.LengthSquared(), Dot(q, q));
        }

        TEST_METHOD(DotOfOrthogonalQuaternionsIsZero) {
            constexpr Quaternion<float> a{ 1.0f, 0.0f, 0.0f, 0.0f };
            constexpr Quaternion<float> b{ 0.0f, 1.0f, 0.0f, 0.0f };

            ExpectNear(0.0f, Dot(a, b));
        }

        TEST_METHOD(LerpAtStart) {
            const Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, 0.0f);
            const Quaternion<float> b = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);

            const Quaternion<float> result = Lerp(a, b, 0.0f);

            ExpectNear(result, a);
        }

        TEST_METHOD(LerpAtEnd) {
            const Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, 0.0f);
            const Quaternion<float> b = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);

            const Quaternion<float> result = Lerp(a, b, 1.0f);

            ExpectNear(result, b);
        }

        TEST_METHOD(LerpIsNormalized) {
            const Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, 0.0f);
            const Quaternion<float> b = Quaternion<float>::FromAxisAngle(Vec3<float>{1.0f, 0.0f, 0.0f}, kPi / 2.0f);

            const Quaternion<float> result = Lerp(a, b, 0.5f);

            ExpectNear(1.0f, result.Length(), 1e-4f);
        }

        TEST_METHOD(SlerpAtStart) {
            const Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, 0.0f);
            const Quaternion<float> b = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);

            const Quaternion<float> result = Slerp(a, b, 0.0f);

            ExpectNear(result, a);
        }

        TEST_METHOD(SlerpAtEnd) {
            const Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, 0.0f);
            const Quaternion<float> b = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);

            const Quaternion<float> result = Slerp(a, b, 1.0f);

            ExpectNear(result, b);
        }

        TEST_METHOD(SlerpAtMidpointMatchesHalfAngle) {
            const Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, 0.0f);
            const Quaternion<float> b = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);
            const Quaternion<float> expected =
                Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, kPi / 4.0f);

            const Quaternion<float> result = Slerp(a, b, 0.5f);

            ExpectNear(result, expected);
        }

        TEST_METHOD(SlerpIsNormalized) {
            const Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, 0.0f);
            const Quaternion<float> b = Quaternion<float>::FromAxisAngle(Vec3<float>{1.0f, 0.0f, 0.0f}, kPi / 2.0f);

            const Quaternion<float> result = Slerp(a, b, 0.3f);

            ExpectNear(1.0f, result.Length(), 1e-4f);
        }

        TEST_METHOD(SlerpTakesShortestPath) {
            const Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, 0.0f);
            Quaternion<float> b = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);
            b = -b; // Negate to represent the same rotation via the opposite hemisphere.

            const Quaternion<float> result = Slerp(a, b, 0.5f);
            const Quaternion<float> expected =
                Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 4.0f);

            ExpectNear(result, expected);
        }

        TEST_METHOD(SlerpNearIdenticalQuaternionsFallsBackToLerp) {
            const Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, 0.0f);
            const Quaternion<float> b = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, 1e-7f);

            const Quaternion<float> result = Slerp(a, b, 0.5f);

            ExpectNear(1.0f, result.Length(), 1e-4f);
        }

        // =========================================================================
        // LerpUnclamped / SlerpUnclamped
        // =========================================================================

        TEST_METHOD(LerpUnclampedMatchesLerpWithinRange) {
            const Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, 0.0f);
            const Quaternion<float> b = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);

            ExpectNear(LerpUnclamped(a, b, 0.5f), Lerp(a, b, 0.5f));
        }

        TEST_METHOD(LerpUnclampedExtrapolatesPastEnd) {
            const Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, 0.0f);
            const Quaternion<float> b = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);

            const Quaternion<float> clamped = Lerp(a, b, 1.5f);
            const Quaternion<float> unclamped = LerpUnclamped(a, b, 1.5f);

            ExpectNear(1.0f, unclamped.Length(), 1e-4f);
            Assert::IsTrue(std::abs(unclamped.Z() - clamped.Z()) > 1e-4f);
        }

        TEST_METHOD(SlerpUnclampedMatchesSlerpWithinRange) {
            const Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, 0.0f);
            const Quaternion<float> b = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);

            ExpectNear(SlerpUnclamped(a, b, 0.5f), Slerp(a, b, 0.5f));
        }

        TEST_METHOD(SlerpUnclampedIsNormalizedPastEnd) {
            const Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, 0.0f);
            const Quaternion<float> b = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);

            const Quaternion<float> result = SlerpUnclamped(a, b, 1.5f);

            ExpectNear(1.0f, result.Length(), 1e-4f);
        }

        // =========================================================================
        // Angle
        // =========================================================================

        TEST_METHOD(AngleBetweenIdenticalQuaternionsIsZero) {
            const Quaternion<float> q = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, kPi / 3.0f);

            ExpectNear(0.0f, Angle(q, q), 1e-4f);
        }

        TEST_METHOD(AngleBetweenIdentityAndRotatedQuaternion) {
            const Quaternion<float> identity = Quaternion<float>::Identity();
            const Quaternion<float> rotated = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);

            ExpectNear(kPi / 2.0f, Angle(identity, rotated), 1e-4f);
        }

        TEST_METHOD(AngleIsSignInvariant) {
            const Quaternion<float> a = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 1.0f, 0.0f}, kPi / 2.0f);
            const Quaternion<float> b = -a;

            // acos loses precision near its domain boundary, so this needs a looser tolerance than usual.
            ExpectNear(0.0f, Angle(a, b), 1e-2f);
        }

        // =========================================================================
        // RotateTowards
        // =========================================================================

        TEST_METHOD(RotateTowardsPartialStep) {
            const Quaternion<float> from = Quaternion<float>::Identity();
            const Quaternion<float> to = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);

            const Quaternion<float> result = RotateTowards(from, to, kPi / 4.0f);

            ExpectNear(kPi / 4.0f, Angle(from, result), 1e-3f);
        }

        TEST_METHOD(RotateTowardsOvershootClampsToTarget) {
            const Quaternion<float> from = Quaternion<float>::Identity();
            const Quaternion<float> to = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 2.0f);

            const Quaternion<float> result = RotateTowards(from, to, kPi);

            ExpectNear(result, to);
        }

        TEST_METHOD(RotateTowardsSameRotationReturnsTarget) {
            const Quaternion<float> from = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 3.0f);
            const Quaternion<float> to = Quaternion<float>::FromAxisAngle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi / 3.0f);

            const Quaternion<float> result = RotateTowards(from, to, kPi / 4.0f);

            ExpectNear(result, to);
        }

        // =========================================================================
        // Aliases
        // =========================================================================

        TEST_METHOD(QuaternionAliases) {
            static_assert(std::is_same_v<Quaternionf, Quaternion<float>>);
            static_assert(std::is_same_v<Quaterniond, Quaternion<double>>);
        }

        // =========================================================================
        // Common type
        // =========================================================================

        TEST_METHOD(CommonType) {
            static_assert(std::is_same_v<QuaternionCommon<float, double>, Quaternion<double>>);
        }

        // =========================================================================
        // Constexpr
        // =========================================================================

        TEST_METHOD(ConstexprConstruction) {
            constexpr Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            static_assert(q.X() == 1.0f);
            static_assert(q.Y() == 2.0f);
            static_assert(q.Z() == 3.0f);
            static_assert(q.W() == 4.0f);
        }

        TEST_METHOD(ConstexprArithmetic) {
            constexpr Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
            constexpr Quaternion<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

            constexpr auto sum = a + b;
            constexpr auto difference = a - b;

            static_assert(sum.X() == 5.0f);
            static_assert(sum.Y() == 7.0f);
            static_assert(sum.Z() == 9.0f);
            static_assert(sum.W() == 11.0f);

            static_assert(difference.X() == -3.0f);
            static_assert(difference.Y() == -3.0f);
            static_assert(difference.Z() == -3.0f);
            static_assert(difference.W() == -3.0f);
        }

        // =========================================================================
        // Formatting
        // =========================================================================

        TEST_METHOD(Formatting) {
            constexpr Quaternion<float> q{ 1.0f, 2.0f, 3.0f, 4.0f };

            Assert::AreEqual(std::string("(1, 2, 3, 4)"), std::format("{}", q));
        }

        TEST_METHOD(FormattingFloatingPoint) {
            constexpr Quaternion<float> q{ 1.0f, 2.5f, 3.25f, 4.0f };

            Assert::AreEqual(std::string("(1, 2.5, 3.25, 4)"), std::format("{}", q));
        }
    };
}
