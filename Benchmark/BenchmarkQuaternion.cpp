#include "Benchmark.h"

#include "Headers/Quaternion.h"
#include "QuaternionSIMD.h"

void BenchmarkQuaternionNoSIMD() {
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

void BenchmarkQuaternionSIMD() {
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
