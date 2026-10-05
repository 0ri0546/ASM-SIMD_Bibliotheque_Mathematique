#include "Benchmark.h"

#include "Headers/Mat.h"
#include "Headers/Quaternion.h"
#include "Headers/Vec.h"

#include "MatSIMD.h"
#include "VecSIMD.h"
#include "QuaternionSIMD.h"

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
        std::vector<Quaternion<float>> q(g_batchSize);
        std::vector<Quaternion<float>> results(g_batchSize);
        const Vec<float, 3> axis{ 0.0f, 1.0f, 0.0f };

        auto result = Benchmark(
            "Quaternionf::FromAxisAngle",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Quaternionf::FromAxisAngle(axis, std::numbers::pi_v<float> / 2.0f);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Quaternion<float>> q(g_batchSize);
        std::vector<Quaternion<float>> results(g_batchSize);
        constexpr Vec<float, 3> v{ 1.0f, 0.0f, 0.0f };

        auto result = Benchmark(
            "Quaternionf::FromToRotation",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Quaternionf::FromToRotation(v, v);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Quaternion<float>> q(g_batchSize);
        std::vector<Quaternion<float>> results(g_batchSize);
        constexpr Vec<float, 3> forward{ 0.0f, 0.0f, 1.0f };
        constexpr Vec<float, 3> up{ 0.0f, 1.0f, 0.0f };

        auto result = Benchmark(
            "Quaternionf::LookRotation",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Quaternion<float>::LookRotation(forward, up);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Quaternion<float>> q(g_batchSize);
        std::vector<Quaternion<float>> results(g_batchSize);
        Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr Quaternion<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "Quaternionf::Addition",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a + b;
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Quaternion<float>> q(g_batchSize);
        std::vector<Quaternion<float>> results(g_batchSize);
        Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr Quaternion<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "Quaternionf::Soustraction",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a - b;
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Quaternion<float>> q(g_batchSize);
        std::vector<Quaternion<float>> results(g_batchSize);
        Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr Quaternion<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "Quaternionf::Multiplication",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a * b;
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Quaternion<float>> q(g_batchSize);
        std::vector<float> results(g_batchSize);
        Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr Quaternion<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "Quaternionf::Dot",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Dot(a, b);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Quaternion<float>> q(g_batchSize);
        std::vector<Quaternion<float>> results(g_batchSize);
        Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr Quaternion<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "Quaternionf::LerpUnclamped",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = LerpUnclamped(a, b, 1.5f);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Quaternion<float>> q(g_batchSize);
        std::vector<Quaternion<float>> results(g_batchSize);
        Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr Quaternion<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "Quaternionf::Lerp",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Lerp(a, b, 1.5f);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Quaternion<float>> q(g_batchSize);
        std::vector<Quaternion<float>> results(g_batchSize);
        Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr Quaternion<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "Quaternionf::SlerpUnclamped",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = SlerpUnclamped(a, b, 1.5f);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Quaternion<float>> q(g_batchSize);
        std::vector<Quaternion<float>> results(g_batchSize);
        Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr Quaternion<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "Quaternionf::Slerp",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Slerp(a, b, 1.5f);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Quaternion<float>> q(g_batchSize);
        std::vector<float> results(g_batchSize);
        Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr Quaternion<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "Quaternionf::Angle",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Angle(a, b);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_referenceResults);
    }

    {
        std::vector<Quaternion<float>> q(g_batchSize);
        std::vector<Quaternion<float>> results(g_batchSize);
        Quaternion<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr Quaternion<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "Quaternionf::RotateTowards",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = RotateTowards(a, b, std::numbers::pi_v<float>);
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

    {
        std::vector<MatSIMD<float, 4, 4>> a4(g_batchSize);
        std::vector<VecSIMD<float, 3>> v3(g_batchSize);
        std::vector<VecSIMD<float, 3>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a4[i] = MatSIMD<float, 4, 4>(
                1.f, 2.f, 3.f, 4.f,
                5.f, 6.f, 7.f, 8.f,
                9.f, 10.f, 11.f, 12.f,
                13.f, 14.f, 15.f, 16.f
            );
            v3[i] = VecSIMD<float, 3>(1.5f, 2.5f, 3.5f);
        }

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::MultiplyPoint",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a4[i].MultiplyPoint(v3[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<MatSIMD<float, 4, 4>> a4(g_batchSize);
        std::vector<MatSIMD<float, 4, 4>> b4(g_batchSize);
        std::vector<MatSIMD<float, 4, 4>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a4[i] = MatSIMD<float, 4, 4>(
                1.f, 2.f, 3.f, 4.f,
                5.f, 6.f, 7.f, 8.f,
                9.f, 10.f, 11.f, 12.f,
                13.f, 14.f, 15.f, 16.f
            );
            b4[i] = MatSIMD<float, 4, 4>(
                1.f, 2.f, 3.f, 4.f,
                5.f, 6.f, 7.f, 8.f,
                9.f, 10.f, 11.f, 12.f,
                13.f, 14.f, 15.f, 16.f
            );
        }

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::Multiplication",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a4[i] * b4[i];
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<MatSIMD<float, 4, 4>> a4(g_batchSize);
        std::vector<VecSIMD<float, 3>> v3(g_batchSize);
        std::vector<VecSIMD<float, 3>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a4[i] = MatSIMD<float, 4, 4>(
                1.f, 2.f, 3.f, 4.f,
                5.f, 6.f, 7.f, 8.f,
                9.f, 10.f, 11.f, 12.f,
                13.f, 14.f, 15.f, 16.f
            );
            v3[i] = VecSIMD<float, 3>(1.5f, 2.5f, 3.5f);
        }

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::MultiplyVector",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a4[i].MultiplyVector(v3[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<MatSIMD<float, 4, 4>> a4(g_batchSize);
        std::vector<MatSIMD<float, 4, 4>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a4[i] = MatSIMD<float, 4, 4>(
                1.f, 2.f, 3.f, 4.f,
                5.f, 6.f, 7.f, 8.f,
                9.f, 10.f, 11.f, 12.f,
                13.f, 14.f, 15.f, 16.f
            );
        }

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::Transpose",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a4[i].Transpose();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<MatSIMD<float, 4, 4>> a4(g_batchSize);
        std::vector<MatSIMD<float, 4, 4>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a4[i] = MatSIMD<float, 4, 4>(
                1.f, 2.f, 3.f, 4.f,
                5.f, 6.f, 7.f, 8.f,
                9.f, 10.f, 11.f, 12.f,
                13.f, 14.f, 15.f, 16.f
            );
        }

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::Inverse",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a4[i].Inverse();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<MatSIMD<float, 4, 4>> a4(g_batchSize);
        std::vector<float> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a4[i] = MatSIMD<float, 4, 4>(
                1.f, 2.f, 3.f, 4.f,
                5.f, 6.f, 7.f, 8.f,
                9.f, 10.f, 11.f, 12.f,
                13.f, 14.f, 15.f, 16.f
            );
        }

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::Determinant",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a4[i].Determinant();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<MatSIMD<float, 4, 4>> results(g_batchSize);

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::Identity",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = MatSIMD<float, 4, 4>::Identity();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<VecSIMD<float, 3>> v3(g_batchSize);
        std::vector<MatSIMD<float, 4, 4>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            v3[i] = VecSIMD<float, 3>(1.5f, 2.5f, 3.5f);
        }

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::Translate",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = MatSIMD<float, 4, 4>::Translate(v3[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<VecSIMD<float, 3>> v3(g_batchSize);
        std::vector<MatSIMD<float, 4, 4>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            v3[i] = VecSIMD<float, 3>(1.5f, 2.5f, 3.5f);
        }

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::Scale",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = MatSIMD<float, 4, 4>::Scale(v3[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<float> angle(g_batchSize, 0.7f);
        std::vector<MatSIMD<float, 4, 4>> results(g_batchSize);

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::RotationX",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = MatSIMD<float, 4, 4>::RotationX(angle[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<float> fov(g_batchSize, 1.0f);
        std::vector<float> aspect(g_batchSize, 1.777f);
        std::vector<float> nearPlane(g_batchSize, 0.1f);
        std::vector<float> farPlane(g_batchSize, 100.f);
        std::vector<MatSIMD<float, 4, 4>> results(g_batchSize);

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::Perspective",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = MatSIMD<float, 4, 4>::Perspective(fov[i], aspect[i], nearPlane[i], farPlane[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<float> fov(g_batchSize, 1.0f);
        std::vector<float> nearPlane(g_batchSize, 0.1f);
        std::vector<float> farPlane(g_batchSize, 100.f);
        std::vector<MatSIMD<float, 4, 4>> results(g_batchSize);

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::Ortho",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = MatSIMD<float, 4, 4>::Ortho(-fov[i], fov[i], -fov[i], fov[i], nearPlane[i], farPlane[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<VecSIMD<float, 3>> eye(g_batchSize);
        std::vector<VecSIMD<float, 3>> target(g_batchSize);
        std::vector<VecSIMD<float, 3>> up(g_batchSize);
        std::vector<MatSIMD<float, 4, 4>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            eye[i] = VecSIMD<float, 3>(0.f, 0.f, 5.f);
            target[i] = VecSIMD<float, 3>(1.5f, 2.5f, 3.5f);
            up[i] = VecSIMD<float, 3>(0.f, 1.f, 0.f);
        }

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::LookAt",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = MatSIMD<float, 4, 4>::LookAt(eye[i], target[i], up[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<VecSIMD<float, 3>> translation(g_batchSize);
        std::vector<QuaternionSIMDf> rotation(g_batchSize);
        std::vector<VecSIMD<float, 3>> scale(g_batchSize);
        std::vector<MatSIMD<float, 4, 4>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            translation[i] = VecSIMD<float, 3>(1.5f, 2.5f, 3.5f);
            rotation[i] = QuaternionSIMDf(1.5f, 2.5f, 3.5f, 4.5f);
            scale[i] = VecSIMD<float, 3>(1.5f, 2.5f, 3.5f);
        }

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::TRS",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = MatSIMD<float, 4, 4>::TRS(translation[i], rotation[i], scale[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<MatSIMD<float, 4, 4>> a4(g_batchSize);
        std::vector<bool> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a4[i] = MatSIMD<float, 4, 4>(
                1.f, 2.f, 3.f, 4.f,
                5.f, 6.f, 7.f, 8.f,
                9.f, 10.f, 11.f, 12.f,
                13.f, 14.f, 15.f, 16.f
            );
        }

        auto result = Benchmark(
            "MatSIMD<float, 4, 4>::ValidTRS",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = ValidTRS(a4[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<MatSIMD<double, 4, 4>> a4d(g_batchSize);
        std::vector<VecSIMD<double, 3>> v3d(g_batchSize);
        std::vector<VecSIMD<double, 3>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a4d[i] = MatSIMD<double, 4, 4>(
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
            );
            v3d[i] = VecSIMD<double, 3>(1.5, 2.5, 3.5);
        }

        auto result = Benchmark(
            "MatSIMD<double, 4, 4>::MultiplyPoint",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a4d[i].MultiplyPoint(v3d[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<MatSIMD<double, 4, 4>> a4(g_batchSize);
        std::vector<MatSIMD<double, 4, 4>> b4(g_batchSize);
        std::vector<MatSIMD<double, 4, 4>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a4[i] = MatSIMD<double, 4, 4>(
                1.f, 2.f, 3.f, 4.f,
                5.f, 6.f, 7.f, 8.f,
                9.f, 10.f, 11.f, 12.f,
                13.f, 14.f, 15.f, 16.f
            );
            b4[i] = MatSIMD<double, 4, 4>(
                1.f, 2.f, 3.f, 4.f,
                5.f, 6.f, 7.f, 8.f,
                9.f, 10.f, 11.f, 12.f,
                13.f, 14.f, 15.f, 16.f
            );
        }

        auto result = Benchmark(
            "MatSIMD<double, 4, 4>::Multiplication",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a4[i] * b4[i];
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<MatSIMD<double, 4, 4>> a4d(g_batchSize);
        std::vector<MatSIMD<double, 4, 4>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a4d[i] = MatSIMD<double, 4, 4>(
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
            );
        }

        auto result = Benchmark(
            "MatSIMD<double, 4, 4>::Transpose",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a4d[i].Transpose();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<MatSIMD<double, 4, 4>> a4d(g_batchSize);
        std::vector<MatSIMD<double, 4, 4>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a4d[i] = MatSIMD<double, 4, 4>(
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
            );
        }

        auto result = Benchmark(
            "MatSIMD<double, 4, 4>::Inverse",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a4d[i].Inverse();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<MatSIMD<double, 4, 4>> a4d(g_batchSize);
        std::vector<double> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            a4d[i] = MatSIMD<double, 4, 4>(
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
            );
        }

        auto result = Benchmark(
            "MatSIMD<double, 4, 4>::Determinant",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a4d[i].Determinant();
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<VecSIMD<double, 3>> translation(g_batchSize);
        std::vector<QuaternionSIMDd> rotation(g_batchSize);
        std::vector<VecSIMD<double, 3>> scale(g_batchSize);
        std::vector<MatSIMD<double, 4, 4>> results(g_batchSize);

        for (std::size_t i = 0; i < g_batchSize; ++i)
        {
            translation[i] = VecSIMD<double, 3>(1.5, 2.5, 3.5);
            rotation[i] = QuaternionSIMDd(1.5, 2.5, 3.5, 4.5);
            scale[i] = VecSIMD<double, 3>(1.5, 2.5, 3.5);
        }

        auto result = Benchmark(
            "MatSIMD<double, 4, 4>::TRS",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = MatSIMD<double, 4, 4>::TRS(translation[i], rotation[i], scale[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }
    {
        std::vector<double> angle(g_batchSize, 0.7);
        std::vector<MatSIMD<double, 4, 4>> results(g_batchSize);

        auto result = Benchmark(
            "MatSIMD<double, 4, 4>::RotationX",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = MatSIMD<double, 4, 4>::RotationX(angle[i]);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<QuaternionSIMD<float>> q(g_batchSize);
        std::vector<QuaternionSIMD<float>> results(g_batchSize);

        auto result = Benchmark(
            "QuaternionSIMD<float>::FromEuler",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = QuaternionSIMD<float>::FromEuler(std::numbers::pi_v<float> / 6.0f, 0.0f, 0.0f);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<QuaternionSIMD<float>> q(g_batchSize);
        std::vector<QuaternionSIMD<float>> results(g_batchSize);
        const VecSIMD<float, 3> axis{ 0.0f, 1.0f, 0.0f };

        auto result = Benchmark(
            "QuaternionSIMD<float>::FromAxisAngle",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = QuaternionSIMD<float>::FromAxisAngle(axis, std::numbers::pi_v<float> / 2.0f);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<QuaternionSIMD<float>> q(g_batchSize);
        std::vector<QuaternionSIMD<float>> results(g_batchSize);
        const VecSIMD<float, 3> v{ 1.0f, 0.0f, 0.0f };

        auto result = Benchmark(
            "QuaternionSIMD<float>::FromToRotation",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = QuaternionSIMD<float>::FromToRotation(v, v);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<QuaternionSIMD<float>> q(g_batchSize);
        std::vector<QuaternionSIMD<float>> results(g_batchSize);
        constexpr VecSIMD<float, 3> forward{ 0.0f, 0.0f, 1.0f };
        constexpr VecSIMD<float, 3> up{ 0.0f, 1.0f, 0.0f };

        auto result = Benchmark(
            "QuaternionSIMD<float>::LookRotation",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = QuaternionSIMD<float>::LookRotation(forward, up);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<QuaternionSIMD<float>> q(g_batchSize);
        std::vector<QuaternionSIMD<float>> results(g_batchSize);
        QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr QuaternionSIMD<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "QuaternionSIMD<float>::Addition",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a + b;
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<QuaternionSIMD<float>> q(g_batchSize);
        std::vector<QuaternionSIMD<float>> results(g_batchSize);
        QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr QuaternionSIMD<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "QuaternionSIMD<float>::Soustraction",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a - b;
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<QuaternionSIMD<float>> q(g_batchSize);
        std::vector<QuaternionSIMD<float>> results(g_batchSize);
        QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr QuaternionSIMD<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "QuaternionSIMD<float>::Multiplication",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = a * b;
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<QuaternionSIMD<float>> q(g_batchSize);
        std::vector<float> results(g_batchSize);
        QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr QuaternionSIMD<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "QuaternionSIMD<float>::Dot",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Dot(a, b);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<QuaternionSIMD<float>> q(g_batchSize);
        std::vector<QuaternionSIMD<float>> results(g_batchSize);
        QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr QuaternionSIMD<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "QuaternionSIMD<float>::LerpUnclamped",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = LerpUnclamped(a, b, 1.5f);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<QuaternionSIMD<float>> q(g_batchSize);
        std::vector<QuaternionSIMD<float>> results(g_batchSize);
        QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr QuaternionSIMD<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "QuaternionSIMD<float>::Lerp",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Lerp(a, b, 1.5f);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<QuaternionSIMD<float>> q(g_batchSize);
        std::vector<QuaternionSIMD<float>> results(g_batchSize);
        QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr QuaternionSIMD<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "QuaternionSIMD<float>::SlerpUnclamped",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = SlerpUnclamped(a, b, 1.5f);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<QuaternionSIMD<float>> q(g_batchSize);
        std::vector<QuaternionSIMD<float>> results(g_batchSize);
        QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr QuaternionSIMD<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "QuaternionSIMD<float>::Slerp",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Slerp(a, b, 1.5f);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<QuaternionSIMD<float>> q(g_batchSize);
        std::vector<float> results(g_batchSize);
        QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr QuaternionSIMD<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "QuaternionSIMD<float>::Angle",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = Angle(a, b);
                }

                DoNotOptimizeAway(results);
            },
            g_batchSize
        );

        PrintBenchmarkResult(result, g_simdResults);
    }

    {
        std::vector<QuaternionSIMD<float>> q(g_batchSize);
        std::vector<QuaternionSIMD<float>> results(g_batchSize);
        QuaternionSIMD<float> a{ 1.0f, 2.0f, 3.0f, 4.0f };
        constexpr QuaternionSIMD<float> b{ 4.0f, 5.0f, 6.0f, 7.0f };

        auto result = Benchmark(
            "QuaternionSIMD<float>::RotateTowards",
            [&]()
            {
                for (std::size_t i = 0; i < g_batchSize; ++i)
                {
                    results[i] = RotateTowards(a, b, std::numbers::pi_v<float>);
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