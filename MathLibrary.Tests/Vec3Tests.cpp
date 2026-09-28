#include "pch.h"
#include "CppUnitTest.h"

#include "Reference/Vec3Reference.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace MathLibraryTests
{
    TEST_CLASS(Vec3Tests)
    {
    public:

        TEST_METHOD(Magnitude)
        {
            Vec3 vector(3.0f, 4.0f, 0.0f);

            float result = vector.magnitude();

            Assert::AreEqual(5.0f, result);
        }

        TEST_METHOD(SqrMagnitude)
        {
            Vec3 vector(3.0f, 4.0f, 0.0f);

            float result = vector.sqrMagnitude();

            Assert::AreEqual(25.0f, result);
        }

        TEST_METHOD(Normalized)
        {
            Vec3 vector(3.0f, 4.0f, 0.0f);

            Vec3 result = vector.normalized();

            Assert::AreEqual(0.6f, result.x, 0.0001f);
            Assert::AreEqual(0.8f, result.y, 0.0001f);
            Assert::AreEqual(0.0f, result.z, 0.0001f);
        }

        TEST_METHOD(NormalizedZero)
        {
            Vec3 vector(0.0f, 0.0f, 0.0f);

            Vec3 result = vector.normalized();

            Assert::AreEqual(0.0f, result.x);
            Assert::AreEqual(0.0f, result.y);
            Assert::AreEqual(0.0f, result.z);
        }

        TEST_METHOD(Dot)
        {
            Vec3 a(1.0f, 2.0f, 3.0f);
            Vec3 b(4.0f, 5.0f, 6.0f);

            float result = Vec3::Dot(a, b);

            Assert::AreEqual(32.0f, result);
        }
    };
}