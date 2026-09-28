#include "pch.h"
#include "CppUnitTest.h"

#include "Reference/BatchOperations.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace MathLibraryTests
{
    TEST_CLASS(BatchTests)
    {
    public:

        TEST_METHOD(DotBatch_Valid)
        {
            Vec3f a[] =
            {
                Vec3f(1.0f, 0.0f, 0.0f),
                Vec3f(0.0f, 1.0f, 0.0f),
                Vec3f(1.0f, 2.0f, 3.0f)
            };

            Vec3f b[] =
            {
                Vec3f(2.0f, 0.0f, 0.0f),
                Vec3f(0.0f, 3.0f, 0.0f),
                Vec3f(4.0f, 5.0f, 6.0f)
            };

            float output[3] = {};

            DotBatch(a, b, output, 3);

            Assert::AreEqual(2.0f, output[0]);
            Assert::AreEqual(3.0f, output[1]);
            Assert::AreEqual(32.0f, output[2]);
        }

        TEST_METHOD(DotBatchEmpty)
        {
            float output[1] = { 123.0f };

            DotBatch(nullptr, nullptr, output, 0);

            Assert::AreEqual(123.0f, output[0]);
        }

        TEST_METHOD(NormalizeBatch_Valid)
        {
            Vec3f input[] =
            {
                Vec3f(3.0f, 4.0f, 0.0f),
                Vec3f(0.0f, 3.0f, 4.0f),
                Vec3f(1.0f, 0.0f, 0.0f)
            };

            Vec3f output[3];

            NormalizeBatch(input, output, 3);

            Assert::AreEqual(0.6f, output[0][0], 0.0001f);
            Assert::AreEqual(0.8f, output[0][1], 0.0001f);
            Assert::AreEqual(0.0f, output[0][2], 0.0001f);

            Assert::AreEqual(0.0f, output[1][0], 0.0001f);
            Assert::AreEqual(0.6f, output[1][1], 0.0001f);
            Assert::AreEqual(0.8f, output[1][2], 0.0001f);

            Assert::AreEqual(1.0f, output[2][0], 0.0001f);
            Assert::AreEqual(0.0f, output[2][1], 0.0001f);
            Assert::AreEqual(0.0f, output[2][2], 0.0001f);
        }

        TEST_METHOD(NormalizeBatchEmpty)
        {
            Vec3f output[1] =
            {
                Vec3f(123.0f, 456.0f, 789.0f)
            };

            NormalizeBatch(nullptr, output, 0);

            Assert::AreEqual(123.0f, output[0][0]);
            Assert::AreEqual(456.0f, output[0][1]);
            Assert::AreEqual(789.0f, output[0][2]);
        }

        TEST_METHOD(TransformPointsBatch_Valid)
        {
            Vec3f input[] =
            {
                Vec3f(1.0f, 2.0f, 3.0f),
                Vec3f(4.0f, 5.0f, 6.0f),
                Vec3f(7.0f, 8.0f, 9.0f)
            };

            Vec3f output[3];

            Mat4f matrix = Mat4f::Translate(
                Vec3f(10.0f, 20.0f, 30.0f)
            );

            TransformPointsBatch(
                input,
                output,
                3,
                matrix
            );

            Assert::AreEqual(11.0f, output[0][0]);
            Assert::AreEqual(22.0f, output[0][1]);
            Assert::AreEqual(33.0f, output[0][2]);

            Assert::AreEqual(14.0f, output[1][0]);
            Assert::AreEqual(25.0f, output[1][1]);
            Assert::AreEqual(36.0f, output[1][2]);

            Assert::AreEqual(17.0f, output[2][0]);
            Assert::AreEqual(28.0f, output[2][1]);
            Assert::AreEqual(39.0f, output[2][2]);
        }

        TEST_METHOD(TransformPointsBatchEmpty)
        {
            Vec3f output[1] =
            {
                Vec3f(123.0f, 456.0f, 789.0f)
            };

            Mat4f matrix = Mat4f::Identity();

            TransformPointsBatch(
                nullptr,
                output,
                0,
                matrix
            );

            Assert::AreEqual(123.0f, output[0][0]);
            Assert::AreEqual(456.0f, output[0][1]);
            Assert::AreEqual(789.0f, output[0][2]);
        }
    };
}