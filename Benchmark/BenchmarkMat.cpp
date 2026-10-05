#include "Benchmark.h"

#include "Headers/Mat.h"
#include "MatSIMD.h"

void BenchmarkMatNoSIMD() {
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
}

void BenchmarkMatSIMD() {
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
}