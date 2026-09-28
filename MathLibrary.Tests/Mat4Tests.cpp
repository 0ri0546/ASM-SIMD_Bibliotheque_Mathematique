#include "pch.h"
#include "CppUnitTest.h"

#include "Reference/Vec3Reference.h"
#include "Reference/Mat4Reference.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace MathLibraryTests
{
    TEST_CLASS(Mat4Tests)
    {
    public:

        TEST_METHOD(IdentityTransform)
        {
            Mat4 matrix = Mat4::Identity();

            Vec3 point(1.0f, 2.0f, 3.0f);

            Vec3 result = matrix.MultiplyPoint3x4(point);

            Assert::AreEqual(1.0f, result.x);
            Assert::AreEqual(2.0f, result.y);
            Assert::AreEqual(3.0f, result.z);
        }

        TEST_METHOD(Translation)
        {
            Mat4 matrix = Mat4::Translate(
                Vec3(10.0f, 20.0f, 30.0f)
            );

            Vec3 point(1.0f, 2.0f, 3.0f);

            Vec3 result = matrix.MultiplyPoint3x4(point);

            Assert::AreEqual(11.0f, result.x);
            Assert::AreEqual(22.0f, result.y);
            Assert::AreEqual(33.0f, result.z);
        }

        TEST_METHOD(Scale)
        {
            Mat4 matrix = Mat4::Scale(
                Vec3(2.0f, 3.0f, 4.0f)
            );

            Vec3 point(1.0f, 2.0f, 3.0f);

            Vec3 result = matrix.MultiplyPoint3x4(point);

            Assert::AreEqual(2.0f, result.x);
            Assert::AreEqual(6.0f, result.y);
            Assert::AreEqual(12.0f, result.z);
        }

        TEST_METHOD(RotationZ90Degrees)
        {
            Mat4 matrix(
                0.0f, -1.0f, 0.0f, 0.0f,
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f
            );

            Vec3 point(1.0f, 0.0f, 0.0f);

            Vec3 result = matrix.MultiplyPoint3x4(point);

            Assert::AreEqual(0.0f, result.x, 0.0001f);
            Assert::AreEqual(1.0f, result.y, 0.0001f);
            Assert::AreEqual(0.0f, result.z, 0.0001f);
        }

        TEST_METHOD(MatrixComposition)
        {
            Mat4 translation = Mat4::Translate(
                Vec3(10.0f, 20.0f, 30.0f)
            );

            Mat4 scale = Mat4::Scale(
                Vec3(2.0f, 2.0f, 2.0f)
            );

            Mat4 transform = translation * scale;

            Vec3 point(1.0f, 2.0f, 3.0f);

            Vec3 result = transform.MultiplyPoint3x4(point);

            Assert::AreEqual(12.0f, result.x);
            Assert::AreEqual(24.0f, result.y);
            Assert::AreEqual(36.0f, result.z);
        }
    };
}