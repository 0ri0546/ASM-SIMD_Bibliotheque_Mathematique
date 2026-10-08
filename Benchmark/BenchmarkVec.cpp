#include "Benchmark.h"

#include "Headers/Vec.h"
#include "VecSIMD.h"

// MASM function
extern "C" void Vec4fAdd(const float* a, const float* b, std::size_t count);

void BenchmarkVecNoSIMD() {
    {
        std::vector<VecSIMD<float, 4>> a(g_batchSize);
        std::vector<VecSIMD<float, 4>> b(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = VecSIMD<float, 4>(
                1.5f,
                2.5f,
                3.5f,
                4.5f
            );
            b[i] = VecSIMD<float, 4>(
                1.5f,
                2.5f,
                3.5f,
                4.5f
            );
        }

        auto result = Benchmark(
            "Vec4fAdd_NoSIMD",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    a[i] += b[i];
                }

                DoNotOptimizeAway(a);
                DoNotOptimizeAway(b);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4f> values(g_batchSize);
        std::vector<Vec4f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            values[i] = Vec4f(
                1.5f,
                2.5f,
                3.5f,
                4.5f
            );
        }

        auto result = Benchmark(
            "Vec4f::Normalize",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = values[i].Normalized();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4f> a(g_batchSize);
        std::vector<Vec4f> b(g_batchSize);
        std::vector<float> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec4f(
                1.0f,
                2.0f,
                3.0f,
                4.0f
            );

            b[i] = Vec4f(
                4.0f,
                5.0f,
                6.0f,
                7.0f
            );
        }

        auto result = Benchmark(
            "Vec4f::Dot",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Dot(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec2d> values(g_batchSize);
        std::vector<Vec2d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            values[i] = Vec2d(
                1.5,
                2.5
            );
        }

        auto result = Benchmark(
            "Vec2d::Normalize",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = values[i].Normalized();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec2d> a(g_batchSize);
        std::vector<Vec2d> b(g_batchSize);
        std::vector<double> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec2d(
                1.0,
                2.0
            );

            b[i] = Vec2d(
                4.0,
                5.0
            );
        }

        auto result = Benchmark(
            "Vec2d::Dot",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Dot(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4d> values(g_batchSize);
        std::vector<Vec4d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            values[i] = Vec4d(
                1.5,
                2.5,
                3.5,
                4.5
            );
        }

        auto result = Benchmark(
            "Vec4d::Normalize",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = values[i].Normalized();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4d> a(g_batchSize);
        std::vector<Vec4d> b(g_batchSize);
        std::vector<double> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec4d(
                1.0,
                2.0,
                3.0,
                4.0
            );

            b[i] = Vec4d(
                4.0,
                5.0,
                6.0,
                7.0
            );
        }

        auto result = Benchmark(
            "Vec4d::Dot",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Dot(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec<float, 8>> values(g_batchSize);
        std::vector<Vec<float, 8>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            values[i] = Vec<float, 8>(
                1.5f,
                2.5f,
                3.5f,
                4.5f,
                5.5f,
                6.5f,
                7.5f,
                8.5f
            );
        }

        auto result = Benchmark(
            "Vec8f::Normalize",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = values[i].Normalized();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec<float, 8>> a(g_batchSize);
        std::vector<Vec<float, 8>> b(g_batchSize);
        std::vector<float> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec<float, 8>(
                1.0f,
                2.0f,
                3.0f,
                4.0f,
                5.0f,
                6.0f,
                7.0f,
                8.0f
            );

            b[i] = Vec<float, 8>(
                4.0f,
                5.0f,
                6.0f,
                7.0f,
                8.0f,
                9.0f,
                10.0f,
                11.0f
            );
        }

        auto result = Benchmark(
            "Vec8f::Dot",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Dot(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec3f> a(g_batchSize);
        std::vector<Vec3f> b(g_batchSize);
        std::vector<Vec3f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec3f(1.0f, 2.0f, 3.0f);
            b[i] = Vec3f(4.0f, 5.0f, 6.0f);
        }

        auto result = Benchmark(
            "Vec3f::Cross",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Cross(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec2f> values(g_batchSize);
        std::vector<Vec2f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            values[i] = Vec2f(1.5f, 2.5f);
        }

        auto result = Benchmark(
            "Vec2f::Perpendicular",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Perpendicular(values[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4f> a(g_batchSize);
        std::vector<Vec4f> b(g_batchSize);
        std::vector<float> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec4f(1.0f, 2.0f, 3.0f, 4.0f);
            b[i] = Vec4f(4.0f, 5.0f, 6.0f, 7.0f);
        }

        auto result = Benchmark(
            "Vec4f::Distance",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Distance(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4f> a(g_batchSize);
        std::vector<Vec4f> b(g_batchSize);
        std::vector<Vec4f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec4f(1.0f, 2.0f, 3.0f, 4.0f);
            b[i] = Vec4f(4.0f, 5.0f, 6.0f, 7.0f);
        }

        auto result = Benchmark(
            "Vec4f::Lerp",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Lerp(a[i], b[i], 0.5f);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4f> a(g_batchSize);
        std::vector<Vec4f> b(g_batchSize);
        std::vector<Vec4f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec4f(1.0f, 2.0f, 3.0f, 4.0f);
            b[i] = Vec4f(4.0f, 5.0f, 6.0f, 7.0f);
        }

        auto result = Benchmark(
            "Vec4f::Scale",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Scale(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4f> a(g_batchSize);
        std::vector<Vec4f> b(g_batchSize);
        std::vector<Vec4f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec4f(1.0f, 5.0f, 3.0f, 7.0f);
            b[i] = Vec4f(4.0f, 2.0f, 6.0f, 4.0f);
        }

        auto result = Benchmark(
            "Vec4f::Min",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Min(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4f> a(g_batchSize);
        std::vector<Vec4f> b(g_batchSize);
        std::vector<Vec4f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec4f(1.0f, 5.0f, 3.0f, 7.0f);
            b[i] = Vec4f(4.0f, 2.0f, 6.0f, 4.0f);
        }

        auto result = Benchmark(
            "Vec4f::Max",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Max(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4f> current(g_batchSize);
        std::vector<Vec4f> target(g_batchSize);
        std::vector<Vec4f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            current[i] = Vec4f(1.0f, 2.0f, 3.0f, 4.0f);
            target[i] = Vec4f(10.0f, 11.0f, 12.0f, 13.0f);
        }

        auto result = Benchmark(
            "Vec4f::MoveTowards",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = MoveTowards(
                        current[i],
                        target[i],
                        2.5f
                    );
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4f> values(g_batchSize);
        std::vector<Vec4f> normals(g_batchSize);
        std::vector<Vec4f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            values[i] = Vec4f(1.0f, -2.0f, 3.0f, -4.0f);
            normals[i] = Vec4f(0.0f, 1.0f, 0.0f, 0.0f);
        }

        auto result = Benchmark(
            "Vec4f::Reflect",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Reflect(
                        values[i],
                        normals[i]
                    );
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4f> a(g_batchSize);
        std::vector<Vec4f> b(g_batchSize);
        std::vector<float> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec4f(1.0f, 2.0f, 3.0f, 4.0f);
            b[i] = Vec4f(4.0f, 5.0f, 6.0f, 7.0f);
        }

        auto result = Benchmark(
            "Vec4f::Angle",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Angle(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec3d> a(g_batchSize);
        std::vector<Vec3d> b(g_batchSize);
        std::vector<Vec3d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec3d(1.0, 2.0, 3.0);
            b[i] = Vec3d(4.0, 5.0, 6.0);
        }

        auto result = Benchmark(
            "Vec3d::Cross",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Cross(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec2d> values(g_batchSize);
        std::vector<Vec2d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            values[i] = Vec2d(1.5, 2.5);
        }

        auto result = Benchmark(
            "Vec2d::Perpendicular",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Perpendicular(values[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4d> a(g_batchSize);
        std::vector<Vec4d> b(g_batchSize);
        std::vector<double> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec4d(1.0, 2.0, 3.0, 4.0);
            b[i] = Vec4d(4.0, 5.0, 6.0, 7.0);
        }

        auto result = Benchmark(
            "Vec4d::Distance",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Distance(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4d> a(g_batchSize);
        std::vector<Vec4d> b(g_batchSize);
        std::vector<Vec4d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec4d(1.0, 2.0, 3.0, 4.0);
            b[i] = Vec4d(4.0, 5.0, 6.0, 7.0);
        }

        auto result = Benchmark(
            "Vec4d::Lerp",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Lerp(a[i], b[i], 0.5);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4d> a(g_batchSize);
        std::vector<Vec4d> b(g_batchSize);
        std::vector<Vec4d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec4d(1.0, 2.0, 3.0, 4.0);
            b[i] = Vec4d(4.0, 5.0, 6.0, 7.0);
        }

        auto result = Benchmark(
            "Vec4d::Scale",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Scale(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4d> a(g_batchSize);
        std::vector<Vec4d> b(g_batchSize);
        std::vector<Vec4d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec4d(1.0, 5.0, 3.0, 7.0);
            b[i] = Vec4d(4.0, 2.0, 6.0, 4.0);
        }

        auto result = Benchmark(
            "Vec4d::Min",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Min(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4d> a(g_batchSize);
        std::vector<Vec4d> b(g_batchSize);
        std::vector<Vec4d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec4d(1.0, 5.0, 3.0, 7.0);
            b[i] = Vec4d(4.0, 2.0, 6.0, 4.0);
        }

        auto result = Benchmark(
            "Vec4d::Max",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Max(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4d> current(g_batchSize);
        std::vector<Vec4d> target(g_batchSize);
        std::vector<Vec4d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            current[i] = Vec4d(1.0, 2.0, 3.0, 4.0);
            target[i] = Vec4d(10.0, 11.0, 12.0, 13.0);
        }

        auto result = Benchmark(
            "Vec4d::MoveTowards",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = MoveTowards(
                        current[i],
                        target[i],
                        2.5
                    );
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4d> values(g_batchSize);
        std::vector<Vec4d> normals(g_batchSize);
        std::vector<Vec4d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            values[i] = Vec4d(1.0, -2.0, 3.0, -4.0);
            normals[i] = Vec4d(0.0, 1.0, 0.0, 0.0);
        }

        auto result = Benchmark(
            "Vec4d::Reflect",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Reflect(
                        values[i],
                        normals[i]
                    );
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec4d> a(g_batchSize);
        std::vector<Vec4d> b(g_batchSize);
        std::vector<double> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = Vec4d(1.0, 2.0, 3.0, 4.0);
            b[i] = Vec4d(4.0, 5.0, 6.0, 7.0);
        }

        auto result = Benchmark(
            "Vec4d::Angle",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Angle(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }
}

void BenchmarkVecSIMD() {
    {
        std::vector<VecSIMD<float, 4>> a(g_batchSize);
        std::vector<VecSIMD<float, 4>> b(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = VecSIMD<float, 4>(
                1.5f,
                2.5f,
                3.5f,
                4.5f
            );
            b[i] = VecSIMD<float, 4>(
                1.5f,
                2.5f,
                3.5f,
                4.5f
            );
        }

        auto result = Benchmark(
            "Vec4fAdd_SIMD",
            [&]()
            {
                Vec4fAdd(a[0].Data(), b[0].Data(), g_batchSize);

                DoNotOptimizeAway(a);
                DoNotOptimizeAway(b);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<VecSIMD<float, 4>> values(g_batchSize);
        std::vector<VecSIMD<float, 4>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            values[i] = VecSIMD<float, 4>(
                1.5f,
                2.5f,
                3.5f,
                4.5f
            );
        }

        auto result = Benchmark(
            "VecSIMD<float, 4>::Normalize",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = values[i].Normalized();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<VecSIMD<float, 4>> a(g_batchSize);
        std::vector<VecSIMD<float, 4>> b(g_batchSize);
        std::vector<float> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = VecSIMD<float, 4>(
                1.0f,
                2.0f,
                3.0f,
                4.0f
            );

            b[i] = VecSIMD<float, 4>(
                4.0f,
                5.0f,
                6.0f,
                7.0f
            );
        }

        auto result = Benchmark(
            "VecSIMD<float, 4>::Dot",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Dot(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<VecSIMD<double, 2>> values(g_batchSize);
        std::vector<VecSIMD<double, 2>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            values[i] = VecSIMD<double, 2>(
                1.5,
                2.5
            );
        }

        auto result = Benchmark(
            "VecSIMD<double, 2>::Normalize",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = values[i].Normalized();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<VecSIMD<double, 2>> a(g_batchSize);
        std::vector<VecSIMD<double, 2>> b(g_batchSize);
        std::vector<double> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = VecSIMD<double, 2>(
                1.0,
                2.0
            );

            b[i] = VecSIMD<double, 2>(
                4.0,
                5.0
            );
        }

        auto result = Benchmark(
            "VecSIMD<double, 2>::Dot",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Dot(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<VecSIMD<double, 4>> values(g_batchSize);
        std::vector<VecSIMD<double, 4>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            values[i] = VecSIMD<double, 4>(
                1.0,
                2.0,
                3.0,
                4.0
            );
        }

        auto result = Benchmark(
            "VecSIMD<double, 4>::Normalize",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = values[i].Normalized();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<VecSIMD<double, 4>> a(g_batchSize);
        std::vector<VecSIMD<double, 4>> b(g_batchSize);
        std::vector<double> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = VecSIMD<double, 4>(
                1.0,
                2.0,
                3.0,
                4.0
            );

            b[i] = VecSIMD<double, 4>(
                4.0,
                5.0,
                6.0,
                7.0
            );
        }

        auto result = Benchmark(
            "VecSIMD<double, 4>::Dot",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Dot(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<VecSIMD<float, 8>> values(g_batchSize);
        std::vector<VecSIMD<float, 8>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            values[i] = VecSIMD<float, 8>(
                1.5f,
                2.5f,
                3.5f,
                4.5f,
                5.5f,
                6.5f,
                7.5f,
                8.5f
            );
        }

        auto result = Benchmark(
            "VecSIMD<float, 8>::Normalize",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = values[i].Normalized();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<VecSIMD<float, 8>> a(g_batchSize);
        std::vector<VecSIMD<float, 8>> b(g_batchSize);
        std::vector<float> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a[i] = VecSIMD<float, 8>(
                1.0f,
                2.0f,
                3.0f,
                4.0f,
                5.0f,
                6.0f,
                7.0f,
                8.0f
            );

            b[i] = VecSIMD<float, 8>(
                4.0f,
                5.0f,
                6.0f,
                7.0f,
                8.0f,
                9.0f,
                10.0f,
                11.0f
            );
        }

        auto result = Benchmark(
            "VecSIMD<float, 8>::Dot",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Dot(a[i], b[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
}
