#include "Benchmark.h"

#include "Headers/Mat.h"
#include "Headers/Quaternion.h"
#include "Headers/Vec.h"
#include "VecSIMD.h"

#include <filesystem>
#include <iostream>

namespace
{
    std::size_t ChooseBatchSize()
    {
        std::println("Choisir la taille du lot :");
        std::println();
        std::println("1. 1 000");
        std::println("2. 100 000");
        std::println("3. 1 000 000");
        std::println("4. Personnalise");
        std::println();

        int choice = 0;

        std::cout << "Choix : ";
        std::cin >> choice;

        if (choice == 1)
            return 1'000;
        if (choice == 2)
            return 100'000;
        if (choice == 3)
            return 1'000'000;
        if (choice == 4) {
            std::size_t size = 0;

            std::cout << "Taille du lot : ";
            std::cin >> size;

            if (size > 0)
                return size;

            std::println("error, on utilise 1000");
            return 1'000;
        }

        std::println("error, on utilise 1000");
        return 1'000;
    }

    int ChooseBenchmarkType()
    {
        std::println();
        std::println("Choisir les benchmarks :");
        std::println();
        std::println("1. Reference C++");
        std::println("2. SIMD");
        std::println("3. Les deux");
        std::println();

        int choice = 0;

        std::cout << "Choix : ";
        std::cin >> choice;

        if (choice >= 1 && choice <= 3)
            return choice;

        std::println("Choix invalide. Les deux seront fait.");
        return 3;
    }
}

void BenchmarkNoSimd()
{
    std::println();
    std::println("Benchmark without SIMD optimizations:");
    PrintBenchmarkHeader();

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

    {
        std::vector<Vec3f> offsets(g_batchSize);
        std::vector<Mat4f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            offsets[i] = Vec3f(10.0f, 20.0f, 30.0f);
        }

        auto result = Benchmark(
            "Mat4f::Translate",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Mat4f::Translate(offsets[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        const Mat4f matrix = Mat4f::Translate(
            Vec3f(10.0f, 20.0f, 30.0f)
        );

        std::vector<Vec3f> points(g_batchSize);
        std::vector<Vec3f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            points[i] = Vec3f(1.0f, 2.0f, 3.0f);
        }

        auto result = Benchmark(
            "Mat4f::MultiplyPoint",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = matrix.MultiplyPoint(points[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec3f> translations(g_batchSize);
        std::vector<Quaternionf> rotations(g_batchSize);
        std::vector<Vec3f> scales(g_batchSize);
        std::vector<Mat4f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            translations[i] = Vec3f(1.0f, 2.0f, 3.0f);
            rotations[i] = Quaternionf::Identity();
            scales[i] = Vec3f(2.0f, 2.0f, 2.0f);
        }

        auto result = Benchmark(
            "Mat4f::TRS",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Mat4f::TRS(
                        translations[i],
                        rotations[i],
                        scales[i]
                    );
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Mat4f> results(g_batchSize);

        auto result = Benchmark(
            "Mat4f::RotationX",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Mat4f::RotationX(0.5f);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        const Mat4f m1 = Mat4f::TRS(
            Vec3f(1.0f, 2.0f, 3.0f),
            Quaternionf::Identity(),
            Vec3f(2.0f, 2.0f, 2.0f)
        );

        const Mat4f m2 = Mat4f::RotationX(0.5f);

        std::vector<Mat4f> results(g_batchSize);

        auto result = Benchmark(
            "Mat4f::Multiplication",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = m1 * m2;
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        const Mat4f matrix = Mat4f::TRS(
            Vec3f(1.0f, 2.0f, 3.0f),
            Quaternionf::Identity(),
            Vec3f(2.0f, 2.0f, 2.0f)
        );

        std::vector<Mat4f> results(g_batchSize);

        auto result = Benchmark(
            "Mat4f::Inverse",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = matrix.Inverse();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Quaternionf> results(g_batchSize);

        auto result = Benchmark(
            "Quaternionf::FromEuler",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Quaternionf::FromEuler(
                        0.5f,
                        0.5f,
                        0.5f
                    );
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        const Quaternionf quaternion = Quaternionf::FromEuler(
            0.5f,
            0.5f,
            0.5f
        );

        std::vector<Vec3f> values(g_batchSize);
        std::vector<Vec3f> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            values[i] = Vec3f(1.0f, 0.0f, 0.0f);
        }

        auto result = Benchmark(
            "Quaternionf::RotateVector",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = quaternion.RotateVector(values[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        const Quaternionf q1 = Quaternionf::Identity();

        const Quaternionf q2 = Quaternionf::FromEuler(
            0.0f,
            1.5f,
            0.0f
        );

        std::vector<Quaternionf> results(g_batchSize);

        auto result = Benchmark(
            "Quaternionf::Slerp",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Slerp(q1, q2, 0.5f);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec3d> offsets(g_batchSize);
        std::vector<Mat4d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            offsets[i] = Vec3d(10.0, 20.0, 30.0);
        }

        auto result = Benchmark(
            "Mat4d::Translate",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Mat4d::Translate(offsets[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        const Mat4d matrix = Mat4d::Translate(
            Vec3d(10.0, 20.0, 30.0)
        );

        std::vector<Vec3d> points(g_batchSize);
        std::vector<Vec3d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            points[i] = Vec3d(1.0, 2.0, 3.0);
        }

        auto result = Benchmark(
            "Mat4d::MultiplyPoint",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = matrix.MultiplyPoint(points[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Vec3d> translations(g_batchSize);
        std::vector<Quaterniond> rotations(g_batchSize);
        std::vector<Vec3d> scales(g_batchSize);
        std::vector<Mat4d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            translations[i] = Vec3d(1.0, 2.0, 3.0);
            rotations[i] = Quaterniond::Identity();
            scales[i] = Vec3d(2.0, 2.0, 2.0);
        }

        auto result = Benchmark(
            "Mat4d::TRS",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Mat4d::TRS(
                        translations[i],
                        rotations[i],
                        scales[i]
                    );
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Mat4d> results(g_batchSize);

        auto result = Benchmark(
            "Mat4d::RotationX",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Mat4d::RotationX(0.5);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        const Mat4d m1 = Mat4d::TRS(
            Vec3d(1.0, 2.0, 3.0),
            Quaterniond::Identity(),
            Vec3d(2.0, 2.0, 2.0)
        );

        const Mat4d m2 = Mat4d::RotationX(0.5);

        std::vector<Mat4d> results(g_batchSize);

        auto result = Benchmark(
            "Mat4d::Multiplication",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = m1 * m2;
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        const Mat4d matrix = Mat4d::TRS(
            Vec3d(1.0, 2.0, 3.0),
            Quaterniond::Identity(),
            Vec3d(2.0, 2.0, 2.0)
        );

        std::vector<Mat4d> results(g_batchSize);

        auto result = Benchmark(
            "Mat4d::Inverse",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = matrix.Inverse();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Quaterniond> results(g_batchSize);

        auto result = Benchmark(
            "Quaterniond::FromEuler",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Quaterniond::FromEuler(
                        0.5,
                        0.5,
                        0.5
                    );
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        const Quaterniond quaternion = Quaterniond::FromEuler(
            0.5,
            0.5,
            0.5
        );

        std::vector<Vec3d> values(g_batchSize);
        std::vector<Vec3d> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            values[i] = Vec3d(1.0, 0.0, 0.0);
        }

        auto result = Benchmark(
            "Quaterniond::RotateVector",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = quaternion.RotateVector(values[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        const Quaterniond q1 = Quaterniond::Identity();

        const Quaterniond q2 = Quaterniond::FromEuler(
            0.0,
            1.5,
            0.0
        );

        std::vector<Quaterniond> results(g_batchSize);

        auto result = Benchmark(
            "Quaterniond::Slerp",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Slerp(q1, q2, 0.5);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }
}

void BenchmarkSIMD()
{
    std::println();
    std::println("Benchmark with SIMD optimizations:");
    PrintBenchmarkHeader();

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

int main()
{
    if (std::filesystem::exists("Benchmark.csv"))
    {
        std::filesystem::remove("Benchmark.csv");
    }

    std::println("==============================================================");
    std::println("                  CONFIGURATION DU BENCHMARK");
    std::println("==============================================================");

    std::size_t batchSize = ChooseBatchSize();
    int benchmarkType = ChooseBenchmarkType();

    constexpr std::uint32_t seed = 0x12345678;

    BeginBenchmark(batchSize, seed);

    if (benchmarkType == 1 || benchmarkType == 3)
    {
        BenchmarkNoSimd();
    }

    if (benchmarkType == 2 || benchmarkType == 3)
    {
        BenchmarkSIMD();
    }

    if (benchmarkType == 3)
    {
        CalculateSpeedups();
        PrintSpeedupTable();
    }

    std::println();
    std::println("Resultats sauvegardes dans Benchmark.csv");

    return 0;
}