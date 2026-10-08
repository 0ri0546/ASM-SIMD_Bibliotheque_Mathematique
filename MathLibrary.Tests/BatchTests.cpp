#include "pch.h"
#include "CppUnitTest.h"

#include <vector>
#include <cmath>

#include "Headers/Batch.h"
#include "BatchSIMD.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace MathLibraryTests
{
    TEST_CLASS(BatchTests)
    {
    private:

        static void AssertNear(
            float expected,
            float actual,
            float epsilon = 0.0001f
        )
        {
            Assert::IsTrue(
                std::abs(expected - actual) < epsilon
            );
        }

        static void AssertVec3Near(
            const Vec3f& expected,
            const Vec3f& actual,
            float epsilon = 0.0001f
        )
        {
            AssertNear(expected[0], actual[0], epsilon);
            AssertNear(expected[1], actual[1], epsilon);
            AssertNear(expected[2], actual[2], epsilon);
        }

    public:

        // ============================================================
        // DOT - AoS
        // ============================================================

        TEST_METHOD(DotBatchAoSTest)
        {
            constexpr std::size_t count = 10;

            std::vector<Vec3f> a(count);
            std::vector<Vec3f> b(count);

            std::vector<float> reference(count);
            std::vector<float> simd(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                a[i] = Vec3f{
                    static_cast<float>(i + 1),
                    static_cast<float>(i + 2),
                    static_cast<float>(i + 3)
                };

                b[i] = Vec3f{
                    2.0f,
                    3.0f,
                    4.0f
                };
            }

            DotBatchAoS(
                a.data(),
                b.data(),
                reference.data(),
                count
            );

            DotBatchSIMDAoS(
                a.data(),
                b.data(),
                simd.data(),
                count
            );

            for (std::size_t i = 0; i < count; ++i)
                AssertNear(reference[i], simd[i]);
        }

        // ============================================================
        // DOT - SoA
        // ============================================================

        TEST_METHOD(DotBatchSoATest)
        {
            constexpr std::size_t count = 10;

            std::vector<float> ax(count);
            std::vector<float> ay(count);
            std::vector<float> az(count);

            std::vector<float> bx(count);
            std::vector<float> by(count);
            std::vector<float> bz(count);

            std::vector<float> reference(count);
            std::vector<float> simd(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                ax[i] = static_cast<float>(i + 1);
                ay[i] = static_cast<float>(i + 2);
                az[i] = static_cast<float>(i + 3);

                bx[i] = 2.0f;
                by[i] = 3.0f;
                bz[i] = 4.0f;
            }

            Vec3Batch<float> a{
                ax.data(),
                ay.data(),
                az.data()
            };

            Vec3Batch<float> b{
                bx.data(),
                by.data(),
                bz.data()
            };

            DotBatchSoA(
                a,
                b,
                reference.data(),
                count
            );

            DotBatchSIMDSoA(
                a,
                b,
                simd.data(),
                count
            );

            for (std::size_t i = 0; i < count; ++i)
                AssertNear(reference[i], simd[i]);
        }

        // ============================================================
        // NORMALIZE - AoS
        // ============================================================

        TEST_METHOD(NormalizeBatchAoSTest)
        {
            constexpr std::size_t count = 10;

            std::vector<Vec3f> input(count);
            std::vector<Vec3f> reference(count);
            std::vector<Vec3f> simd(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                input[i] = Vec3f{
                    static_cast<float>(i + 1),
                    static_cast<float>(i + 2),
                    static_cast<float>(i + 3)
                };
            }

            input[3] = Vec3f{ 0.0f, 0.0f, 0.0f };

            NormalizeBatchAoS(
                input.data(),
                reference.data(),
                count
            );

            NormalizeBatchSIMDAoS(
                input.data(),
                simd.data(),
                count
            );

            for (std::size_t i = 0; i < count; ++i)
                AssertVec3Near(reference[i], simd[i]);
        }

        // ============================================================
        // NORMALIZE - SoA
        // ============================================================

        TEST_METHOD(NormalizeBatchSoATest)
        {
            constexpr std::size_t count = 10;

            std::vector<float> x(count);
            std::vector<float> y(count);
            std::vector<float> z(count);

            std::vector<float> referenceX(count);
            std::vector<float> referenceY(count);
            std::vector<float> referenceZ(count);

            std::vector<float> simdX(count);
            std::vector<float> simdY(count);
            std::vector<float> simdZ(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                x[i] = static_cast<float>(i + 1);
                y[i] = static_cast<float>(i + 2);
                z[i] = static_cast<float>(i + 3);
            }

            x[3] = 0.0f;
            y[3] = 0.0f;
            z[3] = 0.0f;

            Vec3Batch<float> input{
                x.data(),
                y.data(),
                z.data()
            };

            Vec3BatchOutput<float> reference{
                referenceX.data(),
                referenceY.data(),
                referenceZ.data()
            };

            Vec3BatchOutput<float> simd{
                simdX.data(),
                simdY.data(),
                simdZ.data()
            };

            NormalizeBatchSoA(
                input,
                reference,
                count
            );

            NormalizeBatchSIMDSoA(
                input,
                simd,
                count
            );

            for (std::size_t i = 0; i < count; ++i)
            {
                AssertNear(referenceX[i], simdX[i]);
                AssertNear(referenceY[i], simdY[i]);
                AssertNear(referenceZ[i], simdZ[i]);
            }
        }

        // ============================================================
        // TRANSFORM - AoS
        // ============================================================

        TEST_METHOD(TransformPointsBatchAoSTest)
        {
            constexpr std::size_t count = 10;

            std::vector<Vec3f> input(count);
            std::vector<Vec3f> reference(count);
            std::vector<Vec3f> simd(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                input[i] = Vec3f{
                    static_cast<float>(i),
                    static_cast<float>(i + 1),
                    static_cast<float>(i + 2)
                };
            }

            Mat4f matrix = {
                1.0f, 0.0f, 0.0f, 2.0f,
                0.0f, 1.0f, 0.0f, 3.0f,
                0.0f, 0.0f, 1.0f, 4.0f,
                0.0f, 0.0f, 0.0f, 1.0f
            };

            MatSIMD<float, 4, 4> matrixSIMD = {
                1.0f, 0.0f, 0.0f, 2.0f,
                0.0f, 1.0f, 0.0f, 3.0f,
                0.0f, 0.0f, 1.0f, 4.0f,
                0.0f, 0.0f, 0.0f, 1.0f
            };

            TransformPointsBatchAoS(
                matrix,
                input.data(),
                reference.data(),
                count
            );

            TransformPointsBatchSIMDAoS(
                matrixSIMD,
                input.data(),
                simd.data(),
                count
            );

            for (std::size_t i = 0; i < count; ++i)
                AssertVec3Near(reference[i], simd[i]);
        }

        // ============================================================
        // TRANSFORM - SoA
        // ============================================================

        TEST_METHOD(TransformPointsBatchSoATest)
        {
            constexpr std::size_t count = 10;

            std::vector<float> x(count);
            std::vector<float> y(count);
            std::vector<float> z(count);

            std::vector<float> referenceX(count);
            std::vector<float> referenceY(count);
            std::vector<float> referenceZ(count);

            std::vector<float> simdX(count);
            std::vector<float> simdY(count);
            std::vector<float> simdZ(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                x[i] = static_cast<float>(i);
                y[i] = static_cast<float>(i + 1);
                z[i] = static_cast<float>(i + 2);
            }

            Vec3Batch<float> input{
                x.data(),
                y.data(),
                z.data()
            };

            Vec3BatchOutput<float> reference{
                referenceX.data(),
                referenceY.data(),
                referenceZ.data()
            };

            Vec3BatchOutput<float> simd{
                simdX.data(),
                simdY.data(),
                simdZ.data()
            };

            Mat4f matrix = {
                1.0f, 0.0f, 0.0f, 2.0f,
                0.0f, 1.0f, 0.0f, 3.0f,
                0.0f, 0.0f, 1.0f, 4.0f,
                0.0f, 0.0f, 0.0f, 1.0f
            };

            MatSIMD<float, 4, 4> matrixSIMD = {
                1.0f, 0.0f, 0.0f, 2.0f,
                0.0f, 1.0f, 0.0f, 3.0f,
                0.0f, 0.0f, 1.0f, 4.0f,
                0.0f, 0.0f, 0.0f, 1.0f
            };

            TransformPointsBatchSoA(
                matrix,
                input,
                reference,
                count
            );

            TransformPointsBatchSIMDSoA(
                matrixSIMD,
                input,
                simd,
                count
            );

            for (std::size_t i = 0; i < count; ++i)
            {
                AssertNear(referenceX[i], simdX[i]);
                AssertNear(referenceY[i], simdY[i]);
                AssertNear(referenceZ[i], simdZ[i]);
            }
        }

        // ============================================================
        // EMPTY BATCH
        // ============================================================

        TEST_METHOD(BatchEmptyTest)
        {
            constexpr std::size_t count = 0;

            std::vector<Vec3f> input;
            std::vector<Vec3f> output;

            NormalizeBatchAoS(
                input.data(),
                output.data(),
                count
            );

            NormalizeBatchSIMDAoS(
                input.data(),
                output.data(),
                count
            );

            std::vector<float> x;
            std::vector<float> y;
            std::vector<float> z;

            Vec3Batch<float> inputSoA{
                x.data(),
                y.data(),
                z.data()
            };

            Vec3BatchOutput<float> outputSoA{
                x.data(),
                y.data(),
                z.data()
            };

            NormalizeBatchSoA(
                inputSoA,
                outputSoA,
                count
            );

            NormalizeBatchSIMDSoA(
                inputSoA,
                outputSoA,
                count
            );

            Assert::IsTrue(true);
        }

        // ============================================================
        // TAIL - count non multiple of SIMD width
        // ============================================================

        TEST_METHOD(BatchNonMultipleOfSIMDWidthTest)
        {
            constexpr std::size_t count = 13;

            std::vector<Vec3f> a(count);
            std::vector<Vec3f> b(count);

            std::vector<float> reference(count);
            std::vector<float> simd(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                a[i] = Vec3f{
                    static_cast<float>(i + 1),
                    static_cast<float>(i + 2),
                    static_cast<float>(i + 3)
                };

                b[i] = Vec3f{
                    2.0f,
                    3.0f,
                    4.0f
                };
            }

            DotBatchAoS(
                a.data(),
                b.data(),
                reference.data(),
                count
            );

            DotBatchSIMDAoS(
                a.data(),
                b.data(),
                simd.data(),
                count
            );

            for (std::size_t i = 0; i < count; ++i)
                AssertNear(reference[i], simd[i]);
        }

        // ============================================================
        // CROSS CHECK AoS / SoA - DOT
        // ============================================================

        TEST_METHOD(DotBatchAoSAndSoATest)
        {
            constexpr std::size_t count = 13;

            std::vector<Vec3f> a(count);
            std::vector<Vec3f> b(count);

            std::vector<float> aos(count);
            std::vector<float> soa(count);

            std::vector<float> ax(count);
            std::vector<float> ay(count);
            std::vector<float> az(count);

            std::vector<float> bx(count);
            std::vector<float> by(count);
            std::vector<float> bz(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                a[i] = Vec3f{
                    static_cast<float>(i + 1),
                    static_cast<float>(i + 2),
                    static_cast<float>(i + 3)
                };

                b[i] = Vec3f{
                    2.0f,
                    3.0f,
                    4.0f
                };

                ax[i] = a[i][0];
                ay[i] = a[i][1];
                az[i] = a[i][2];

                bx[i] = b[i][0];
                by[i] = b[i][1];
                bz[i] = b[i][2];
            }

            Vec3Batch<float> aBatch{
                ax.data(),
                ay.data(),
                az.data()
            };

            Vec3Batch<float> bBatch{
                bx.data(),
                by.data(),
                bz.data()
            };

            DotBatchAoS(
                a.data(),
                b.data(),
                aos.data(),
                count
            );

            DotBatchSIMDSoA(
                aBatch,
                bBatch,
                soa.data(),
                count
            );

            for (std::size_t i = 0; i < count; ++i)
                AssertNear(aos[i], soa[i]);
        }

        // ============================================================
        // CROSS CHECK AoS / SoA - NORMALIZE
        // ============================================================

        TEST_METHOD(NormalizeBatchAoSAndSoATest)
        {
            constexpr std::size_t count = 13;

            std::vector<Vec3f> input(count);
            std::vector<Vec3f> aos(count);

            std::vector<float> x(count);
            std::vector<float> y(count);
            std::vector<float> z(count);

            std::vector<float> soaX(count);
            std::vector<float> soaY(count);
            std::vector<float> soaZ(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                input[i] = Vec3f{
                    static_cast<float>(i + 1),
                    static_cast<float>(i + 2),
                    static_cast<float>(i + 3)
                };

                x[i] = input[i][0];
                y[i] = input[i][1];
                z[i] = input[i][2];
            }

            Vec3Batch<float> inputBatch{
                x.data(),
                y.data(),
                z.data()
            };

            Vec3BatchOutput<float> outputBatch{
                soaX.data(),
                soaY.data(),
                soaZ.data()
            };

            NormalizeBatchAoS(
                input.data(),
                aos.data(),
                count
            );

            NormalizeBatchSIMDSoA(
                inputBatch,
                outputBatch,
                count
            );

            for (std::size_t i = 0; i < count; ++i)
            {
                AssertNear(aos[i][0], soaX[i]);
                AssertNear(aos[i][1], soaY[i]);
                AssertNear(aos[i][2], soaZ[i]);
            }
        }

        // ============================================================
        // CROSS CHECK AoS / SoA - TRANSFORM
        // ============================================================

        TEST_METHOD(TransformPointsBatchAoSAndSoATest)
        {
            constexpr std::size_t count = 13;

            std::vector<Vec3f> input(count);
            std::vector<Vec3f> aos(count);

            std::vector<float> x(count);
            std::vector<float> y(count);
            std::vector<float> z(count);

            std::vector<float> soaX(count);
            std::vector<float> soaY(count);
            std::vector<float> soaZ(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                input[i] = Vec3f{
                    static_cast<float>(i),
                    static_cast<float>(i + 1),
                    static_cast<float>(i + 2)
                };

                x[i] = input[i][0];
                y[i] = input[i][1];
                z[i] = input[i][2];
            }

            Vec3Batch<float> inputBatch{
                x.data(),
                y.data(),
                z.data()
            };

            Vec3BatchOutput<float> outputBatch{
                soaX.data(),
                soaY.data(),
                soaZ.data()
            };

            Mat4f matrix = {
                1.0f, 0.0f, 0.0f, 2.0f,
                0.0f, 1.0f, 0.0f, 3.0f,
                0.0f, 0.0f, 1.0f, 4.0f,
                0.0f, 0.0f, 0.0f, 1.0f
            };

            TransformPointsBatchAoS(
                matrix,
                input.data(),
                aos.data(),
                count
            );

            TransformPointsBatchSoA(
                matrix,
                inputBatch,
                outputBatch,
                count
            );

            for (std::size_t i = 0; i < count; ++i)
            {
                AssertNear(aos[i][0], soaX[i]);
                AssertNear(aos[i][1], soaY[i]);
                AssertNear(aos[i][2], soaZ[i]);
            }
        }
    };
}