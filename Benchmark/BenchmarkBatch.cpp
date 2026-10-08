#include "Benchmark.h"

#include "Headers/Batch.h"
#include "BatchSIMD.h"

// ============================================================
// Reference
// ============================================================

void BenchmarkBatchNoSIMD()
{
    // ========================================================
    // Dot - AoS - float
    // ========================================================

    {
        std::vector<Vec3f> a(g_batchSize);
        std::vector<Vec3f> b(g_batchSize);
        std::vector<float> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec3f(1.0f, 2.0f, 3.0f);
            b[i] = Vec3f(4.0f, 5.0f, 6.0f);
        }

        auto result = Benchmark(
            "Vec3f::DotBatch_AoS",
            [&]()
            {
                DotBatchAoS(
                    a.data(),
                    b.data(),
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    // ========================================================
    // Dot - AoS - double
    // ========================================================

    {
        std::vector<Vec3d> a(g_batchSize);
        std::vector<Vec3d> b(g_batchSize);
        std::vector<double> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec3d(1.0, 2.0, 3.0);
            b[i] = Vec3d(4.0, 5.0, 6.0);
        }

        auto result = Benchmark(
            "Vec3d::DotBatch_AoS",
            [&]()
            {
                DotBatchAoS(
                    a.data(),
                    b.data(),
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    // ========================================================
    // Dot - SoA - float
    // ========================================================

    {
        std::vector<float> ax(g_batchSize);
        std::vector<float> ay(g_batchSize);
        std::vector<float> az(g_batchSize);

        std::vector<float> bx(g_batchSize);
        std::vector<float> by(g_batchSize);
        std::vector<float> bz(g_batchSize);

        std::vector<float> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            ax[i] = 1.0f;
            ay[i] = 2.0f;
            az[i] = 3.0f;

            bx[i] = 4.0f;
            by[i] = 5.0f;
            bz[i] = 6.0f;
        }

        const Vec3Batch<float> a{
            ax.data(),
            ay.data(),
            az.data()
        };

        const Vec3Batch<float> b{
            bx.data(),
            by.data(),
            bz.data()
        };

        auto result = Benchmark(
            "Vec3f::DotBatch_SoA",
            [&]()
            {
                DotBatchSoA(
                    a,
                    b,
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    // ========================================================
    // Dot - SoA - double
    // ========================================================

    {
        std::vector<double> ax(g_batchSize);
        std::vector<double> ay(g_batchSize);
        std::vector<double> az(g_batchSize);

        std::vector<double> bx(g_batchSize);
        std::vector<double> by(g_batchSize);
        std::vector<double> bz(g_batchSize);

        std::vector<double> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            ax[i] = 1.0;
            ay[i] = 2.0;
            az[i] = 3.0;

            bx[i] = 4.0;
            by[i] = 5.0;
            bz[i] = 6.0;
        }

        const Vec3Batch<double> a{
            ax.data(),
            ay.data(),
            az.data()
        };

        const Vec3Batch<double> b{
            bx.data(),
            by.data(),
            bz.data()
        };

        auto result = Benchmark(
            "Vec3d::DotBatch_SoA",
            [&]()
            {
                DotBatchSoA(
                    a,
                    b,
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    // ========================================================
    // Normalize - AoS - float
    // ========================================================

    {
        std::vector<Vec3f> input(g_batchSize);
        std::vector<Vec3f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
            input[i] = Vec3f(1.0f, 2.0f, 3.0f);

        auto result = Benchmark(
            "Vec3f::NormalizeBatch_AoS",
            [&]()
            {
                NormalizeBatchAoS(
                    input.data(),
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    // ========================================================
    // Normalize - AoS - double
    // ========================================================

    {
        std::vector<Vec3d> input(g_batchSize);
        std::vector<Vec3d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
            input[i] = Vec3d(1.0, 2.0, 3.0);

        auto result = Benchmark(
            "Vec3d::NormalizeBatch_AoS",
            [&]()
            {
                NormalizeBatchAoS(
                    input.data(),
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    // ========================================================
    // Normalize - SoA - float
    // ========================================================

    {
        std::vector<float> x(g_batchSize);
        std::vector<float> y(g_batchSize);
        std::vector<float> z(g_batchSize);

        std::vector<float> outX(g_batchSize);
        std::vector<float> outY(g_batchSize);
        std::vector<float> outZ(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            x[i] = 1.0f;
            y[i] = 2.0f;
            z[i] = 3.0f;
        }

        const Vec3Batch<float> input{
            x.data(),
            y.data(),
            z.data()
        };

        const Vec3BatchOutput<float> output{
            outX.data(),
            outY.data(),
            outZ.data()
        };

        auto result = Benchmark(
            "Vec3f::NormalizeBatch_SoA",
            [&]()
            {
                NormalizeBatchSoA(
                    input,
                    output,
                    g_batchSize
                );

                DoNotOptimizeAway(outX);
                DoNotOptimizeAway(outY);
                DoNotOptimizeAway(outZ);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    // ========================================================
    // Normalize - SoA - double
    // ========================================================

    {
        std::vector<double> x(g_batchSize);
        std::vector<double> y(g_batchSize);
        std::vector<double> z(g_batchSize);

        std::vector<double> outX(g_batchSize);
        std::vector<double> outY(g_batchSize);
        std::vector<double> outZ(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            x[i] = 1.0;
            y[i] = 2.0;
            z[i] = 3.0;
        }

        const Vec3Batch<double> input{
            x.data(),
            y.data(),
            z.data()
        };

        const Vec3BatchOutput<double> output{
            outX.data(),
            outY.data(),
            outZ.data()
        };

        auto result = Benchmark(
            "Vec3d::NormalizeBatch_SoA",
            [&]()
            {
                NormalizeBatchSoA(
                    input,
                    output,
                    g_batchSize
                );

                DoNotOptimizeAway(outX);
                DoNotOptimizeAway(outY);
                DoNotOptimizeAway(outZ);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    // ========================================================
    // Transform - AoS - float
    // ========================================================

    {
        const Mat4f matrix = {
            1.0f, 0.0f, 0.0f, 2.0f,
            0.0f, 1.0f, 0.0f, 3.0f,
            0.0f, 0.0f, 1.0f, 4.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };

        std::vector<Vec3f> input(g_batchSize);
        std::vector<Vec3f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
            input[i] = Vec3f(1.0f, 2.0f, 3.0f);

        auto result = Benchmark(
            "Mat4f::TransformPointsBatch_AoS",
            [&]()
            {
                TransformPointsBatchAoS(
                    matrix,
                    input.data(),
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    // ========================================================
    // Transform - AoS - double
    // ========================================================

    {
        const Mat4d matrix = {
            1.0, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, 3.0,
            0.0, 0.0, 1.0, 4.0,
            0.0, 0.0, 0.0, 1.0
        };

        std::vector<Vec3d> input(g_batchSize);
        std::vector<Vec3d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
            input[i] = Vec3d(1.0, 2.0, 3.0);

        auto result = Benchmark(
            "Mat4d::TransformPointsBatch_AoS",
            [&]()
            {
                TransformPointsBatchAoS(
                    matrix,
                    input.data(),
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    // ========================================================
    // Transform - SoA - float
    // ========================================================

    {
        const Mat4f matrix = {
            1.0f, 0.0f, 0.0f, 2.0f,
            0.0f, 1.0f, 0.0f, 3.0f,
            0.0f, 0.0f, 1.0f, 4.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };

        std::vector<float> x(g_batchSize);
        std::vector<float> y(g_batchSize);
        std::vector<float> z(g_batchSize);

        std::vector<float> outX(g_batchSize);
        std::vector<float> outY(g_batchSize);
        std::vector<float> outZ(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            x[i] = 1.0f;
            y[i] = 2.0f;
            z[i] = 3.0f;
        }

        const Vec3Batch<float> input{
            x.data(),
            y.data(),
            z.data()
        };

        const Vec3BatchOutput<float> output{
            outX.data(),
            outY.data(),
            outZ.data()
        };

        auto result = Benchmark(
            "Mat4f::TransformPointsBatch_SoA",
            [&]()
            {
                TransformPointsBatchSoA(
                    matrix,
                    input,
                    output,
                    g_batchSize
                );

                DoNotOptimizeAway(outX);
                DoNotOptimizeAway(outY);
                DoNotOptimizeAway(outZ);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    // ========================================================
    // Transform - SoA - double
    // ========================================================

    {
        const Mat4d matrix = {
            1.0, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, 3.0,
            0.0, 0.0, 1.0, 4.0,
            0.0, 0.0, 0.0, 1.0
        };

        std::vector<double> x(g_batchSize);
        std::vector<double> y(g_batchSize);
        std::vector<double> z(g_batchSize);

        std::vector<double> outX(g_batchSize);
        std::vector<double> outY(g_batchSize);
        std::vector<double> outZ(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            x[i] = 1.0;
            y[i] = 2.0;
            z[i] = 3.0;
        }

        const Vec3Batch<double> input{
            x.data(),
            y.data(),
            z.data()
        };

        const Vec3BatchOutput<double> output{
            outX.data(),
            outY.data(),
            outZ.data()
        };

        auto result = Benchmark(
            "Mat4d::TransformPointsBatch_SoA",
            [&]()
            {
                TransformPointsBatchSoA(
                    matrix,
                    input,
                    output,
                    g_batchSize
                );

                DoNotOptimizeAway(outX);
                DoNotOptimizeAway(outY);
                DoNotOptimizeAway(outZ);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }
}

// ============================================================
// SIMD
// ============================================================

void BenchmarkBatchSIMD()
{
    // ========================================================
    // Dot - AoS - float
    // ========================================================

    {
        std::vector<Vec3f> a(g_batchSize);
        std::vector<Vec3f> b(g_batchSize);
        std::vector<float> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec3f(1.0f, 2.0f, 3.0f);
            b[i] = Vec3f(4.0f, 5.0f, 6.0f);
        }

        auto result = Benchmark(
            "VecSIMD<float, 3>::DotBatch_AoS",
            [&]()
            {
                DotBatchSIMDAoS(
                    a.data(),
                    b.data(),
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    // ========================================================
    // Dot - AoS - double
    // ========================================================

    {
        std::vector<Vec3d> a(g_batchSize);
        std::vector<Vec3d> b(g_batchSize);
        std::vector<double> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec3d(1.0, 2.0, 3.0);
            b[i] = Vec3d(4.0, 5.0, 6.0);
        }

        auto result = Benchmark(
            "VecSIMD<double, 3>::DotBatch_AoS",
            [&]()
            {
                DotBatchSIMDAoS(
                    a.data(),
                    b.data(),
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    // ========================================================
    // Dot - SoA - float
    // ========================================================

    {
        std::vector<float> ax(g_batchSize);
        std::vector<float> ay(g_batchSize);
        std::vector<float> az(g_batchSize);

        std::vector<float> bx(g_batchSize);
        std::vector<float> by(g_batchSize);
        std::vector<float> bz(g_batchSize);

        std::vector<float> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            ax[i] = 1.0f;
            ay[i] = 2.0f;
            az[i] = 3.0f;

            bx[i] = 4.0f;
            by[i] = 5.0f;
            bz[i] = 6.0f;
        }

        const Vec3Batch<float> a{
            ax.data(),
            ay.data(),
            az.data()
        };

        const Vec3Batch<float> b{
            bx.data(),
            by.data(),
            bz.data()
        };

        auto result = Benchmark(
            "VecSIMD<float, 3>::DotBatch_SoA",
            [&]()
            {
                DotBatchSIMDSoA(
                    a,
                    b,
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    // ========================================================
    // Dot - SoA - double
    // ========================================================

    {
        std::vector<double> ax(g_batchSize);
        std::vector<double> ay(g_batchSize);
        std::vector<double> az(g_batchSize);

        std::vector<double> bx(g_batchSize);
        std::vector<double> by(g_batchSize);
        std::vector<double> bz(g_batchSize);

        std::vector<double> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            ax[i] = 1.0;
            ay[i] = 2.0;
            az[i] = 3.0;

            bx[i] = 4.0;
            by[i] = 5.0;
            bz[i] = 6.0;
        }

        const Vec3Batch<double> a{
            ax.data(),
            ay.data(),
            az.data()
        };

        const Vec3Batch<double> b{
            bx.data(),
            by.data(),
            bz.data()
        };

        auto result = Benchmark(
            "VecSIMD<double, 3>::DotBatch_SoA",
            [&]()
            {
                DotBatchSIMDSoA(
                    a,
                    b,
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    // ========================================================
    // Normalize - AoS - float
    // ========================================================

    {
        std::vector<Vec3f> input(g_batchSize);
        std::vector<Vec3f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
            input[i] = Vec3f(1.0f, 2.0f, 3.0f);

        auto result = Benchmark(
            "VecSIMD<float, 3>::NormalizeBatch_AoS",
            [&]()
            {
                NormalizeBatchSIMDAoS(
                    input.data(),
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    // ========================================================
    // Normalize - AoS - double
    // ========================================================

    {
        std::vector<Vec3d> input(g_batchSize);
        std::vector<Vec3d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
            input[i] = Vec3d(1.0, 2.0, 3.0);

        auto result = Benchmark(
            "VecSIMD<double, 3>::NormalizeBatch_AoS",
            [&]()
            {
                NormalizeBatchSIMDAoS(
                    input.data(),
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    // ========================================================
    // Normalize - SoA - float
    // ========================================================

    {
        std::vector<float> x(g_batchSize);
        std::vector<float> y(g_batchSize);
        std::vector<float> z(g_batchSize);

        std::vector<float> outX(g_batchSize);
        std::vector<float> outY(g_batchSize);
        std::vector<float> outZ(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            x[i] = 1.0f;
            y[i] = 2.0f;
            z[i] = 3.0f;
        }

        const Vec3Batch<float> input{
            x.data(),
            y.data(),
            z.data()
        };

        const Vec3BatchOutput<float> output{
            outX.data(),
            outY.data(),
            outZ.data()
        };

        auto result = Benchmark(
            "VecSIMD<float, 3>::NormalizeBatch_SoA",
            [&]()
            {
                NormalizeBatchSIMDSoA(
                    input,
                    output,
                    g_batchSize
                );

                DoNotOptimizeAway(outX);
                DoNotOptimizeAway(outY);
                DoNotOptimizeAway(outZ);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    // ========================================================
    // Normalize - SoA - double
    // ========================================================

    {
        std::vector<double> x(g_batchSize);
        std::vector<double> y(g_batchSize);
        std::vector<double> z(g_batchSize);

        std::vector<double> outX(g_batchSize);
        std::vector<double> outY(g_batchSize);
        std::vector<double> outZ(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            x[i] = 1.0;
            y[i] = 2.0;
            z[i] = 3.0;
        }

        const Vec3Batch<double> input{
            x.data(),
            y.data(),
            z.data()
        };

        const Vec3BatchOutput<double> output{
            outX.data(),
            outY.data(),
            outZ.data()
        };

        auto result = Benchmark(
            "VecSIMD<double, 3>::NormalizeBatch_SoA",
            [&]()
            {
                NormalizeBatchSIMDSoA(
                    input,
                    output,
                    g_batchSize
                );

                DoNotOptimizeAway(outX);
                DoNotOptimizeAway(outY);
                DoNotOptimizeAway(outZ);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    // ========================================================
    // Transform - AoS - float
    // ========================================================

    {
        const MatSIMD<float, 4, 4> matrix = {
            1.0f, 0.0f, 0.0f, 2.0f,
            0.0f, 1.0f, 0.0f, 3.0f,
            0.0f, 0.0f, 1.0f, 4.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };

        std::vector<Vec3f> input(g_batchSize);
        std::vector<Vec3f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
            input[i] = Vec3f(1.0f, 2.0f, 3.0f);

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::TransformPointsBatch_AoS",
            [&]()
            {
                TransformPointsBatchSIMDAoS(
                    matrix,
                    input.data(),
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    // ========================================================
    // Transform - AoS - double
    // ========================================================

    {
        const MatSIMD<double, 4, 4> matrix = {
            1.0, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, 3.0,
            0.0, 0.0, 1.0, 4.0,
            0.0, 0.0, 0.0, 1.0
        };

        std::vector<Vec3d> input(g_batchSize);
        std::vector<Vec3d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
            input[i] = Vec3d(1.0, 2.0, 3.0);

        auto result = Benchmark(
            "MatSIMD<double, 4, 4>::TransformPointsBatch_AoS",
            [&]()
            {
                TransformPointsBatchSIMDAoS(
                    matrix,
                    input.data(),
                    results.data(),
                    g_batchSize
                );

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    // ========================================================
    // Transform - SoA - float
    // ========================================================

    {
        const MatSIMD<float, 4, 4> matrix = {
            1.0f, 0.0f, 0.0f, 2.0f,
            0.0f, 1.0f, 0.0f, 3.0f,
            0.0f, 0.0f, 1.0f, 4.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };

        std::vector<float> x(g_batchSize);
        std::vector<float> y(g_batchSize);
        std::vector<float> z(g_batchSize);

        std::vector<float> outX(g_batchSize);
        std::vector<float> outY(g_batchSize);
        std::vector<float> outZ(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            x[i] = 1.0f;
            y[i] = 2.0f;
            z[i] = 3.0f;
        }

        const Vec3Batch<float> input{
            x.data(),
            y.data(),
            z.data()
        };

        const Vec3BatchOutput<float> output{
            outX.data(),
            outY.data(),
            outZ.data()
        };

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::TransformPointsBatch_SoA",
            [&]()
            {
                TransformPointsBatchSIMDSoA(
                    matrix,
                    input,
                    output,
                    g_batchSize
                );

                DoNotOptimizeAway(outX);
                DoNotOptimizeAway(outY);
                DoNotOptimizeAway(outZ);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    // ========================================================
    // Transform - SoA - double
    // ========================================================

    {
        const MatSIMD<double, 4, 4> matrix = {
            1.0, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, 3.0,
            0.0, 0.0, 1.0, 4.0,
            0.0, 0.0, 0.0, 1.0
        };

        std::vector<double> x(g_batchSize);
        std::vector<double> y(g_batchSize);
        std::vector<double> z(g_batchSize);

        std::vector<double> outX(g_batchSize);
        std::vector<double> outY(g_batchSize);
        std::vector<double> outZ(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            x[i] = 1.0;
            y[i] = 2.0;
            z[i] = 3.0;
        }

        const Vec3Batch<double> input{
            x.data(),
            y.data(),
            z.data()
        };

        const Vec3BatchOutput<double> output{
            outX.data(),
            outY.data(),
            outZ.data()
        };

        auto result = Benchmark(
            "MatSIMD<double, 4, 4>::TransformPointsBatch_SoA",
            [&]()
            {
                TransformPointsBatchSIMDSoA(
                    matrix,
                    input,
                    output,
                    g_batchSize
                );

                DoNotOptimizeAway(outX);
                DoNotOptimizeAway(outY);
                DoNotOptimizeAway(outZ);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
}