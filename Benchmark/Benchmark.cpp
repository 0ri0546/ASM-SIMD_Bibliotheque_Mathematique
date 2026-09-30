#include "Benchmark.h"

#include "Headers/Mat.h"
#include "Headers/Quaternion.h"
#include "Headers/Vec.h"

#include "VecSIMD.h"

void BenchmarkNoSimd() {
    std::println("Benchmark without SIMD optimizations:");
    std::println("{:-<122}", "");
    std::println("{:<4} | {:<40} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12}", "No.", "Name", "op/s", "ns/op", "err%", "cyc/op", "total (ms)");
    std::println("{:-<122}", "");

    Benchmark("Vec3f::Normalize", []() {
        Vec3f v(1.5f, 2.5f, 3.5f);
        Vec3f res = v.Normalized();
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec3f::DotAndCross", []() {
        Vec3f a(1.0f, 2.0f, 3.0f);
        Vec3f b(4.0f, 5.0f, 6.0f);
        float d = Dot(a, b);
        Vec3f c = Cross(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(d);
        DoNotOptimizeAway(c);
        });

    Benchmark("Mat4f::Translate & MultiplyPoint", []() {
        Vec3f point(1.0f, 2.0f, 3.0f);
        Vec3f offset(10.0f, 20.0f, 30.0f);
        Mat4f translation = Mat4f::Translate(offset);
        Vec3f result = translation.MultiplyPoint(point);
        DoNotOptimizeAway(point);
        DoNotOptimizeAway(offset);
        DoNotOptimizeAway(result);
        });

    Benchmark("Mat4f::TRS & Multiplication", []() {
        Vec3f t(1.f, 2.f, 3.f);
        Quaternionf r = Quaternionf::Identity();
        Vec3f s(2.f, 2.f, 2.f);
        float angle = 0.5f;
        Mat4f m1 = Mat4f::TRS(t, r, s);
        Mat4f m2 = Mat4f::RotationX(angle);
        Mat4f result = m1 * m2;
        DoNotOptimizeAway(t);
        DoNotOptimizeAway(r);
        DoNotOptimizeAway(s);
        DoNotOptimizeAway(angle);
        DoNotOptimizeAway(result);
        });

    Benchmark("Mat4f::Inverse", []() {
        Mat4f inv = Mat4f().Inverse();
        DoNotOptimizeAway(inv);
        });

    Benchmark("Quaternionf::FromEuler & RotateVector", []() {
        float angle = 0.5f;
        Vec3f v(1.0f, 0.0f, 0.0f);
        Quaternionf q = Quaternionf::FromEuler(angle, angle, angle);
        Vec3f result = q.RotateVector(v);
        DoNotOptimizeAway(angle);
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(result);
        });

    Benchmark("Quaternionf::Slerp", []() {
        Quaternionf q1 = Quaternionf::Identity();
        float angle = 1.5f;
        float t = 0.5f;
        Quaternionf q2 = Quaternionf::FromEuler(0.0f, angle, 0.0f);
        Quaternionf result = Slerp(q1, q2, t);
        DoNotOptimizeAway(q1);
        DoNotOptimizeAway(angle);
        DoNotOptimizeAway(t);
        DoNotOptimizeAway(result);
        });

    Benchmark("Vec3d::Normalize", []() {
        Vec3d v(1.5, 2.5, 3.5);
        Vec3d res = v.Normalized();
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec3d::DotAndCross", []() {
        Vec3d a(1.0, 2.0, 3.0);
        Vec3d b(4.0, 5.0, 6.0);
        double d = Dot(a, b);
        Vec3d c = Cross(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(d);
        DoNotOptimizeAway(c);
        });

    Benchmark("Mat4d::Translate & MultiplyPoint", []() {
        Vec3d point(1.0, 2.0, 3.0);
        Vec3d offset(10.0, 20.0, 30.0);
        Mat4d translation = Mat4d::Translate(offset);
        Vec3d result = translation.MultiplyPoint(point);
        DoNotOptimizeAway(point);
        DoNotOptimizeAway(offset);
        DoNotOptimizeAway(result);
        });

    Benchmark("Mat4d::TRS & Multiplication", []() {
        Vec3d t(1.0, 2.0, 3.0);
        Quaterniond r = Quaterniond::Identity();
        Vec3d s(2.0, 2.0, 2.0);
        double angle = 0.5;
        Mat4d m1 = Mat4d::TRS(t, r, s);
        Mat4d m2 = Mat4d::RotationX(angle);
        Mat4d result = m1 * m2;
        DoNotOptimizeAway(t);
        DoNotOptimizeAway(r);
        DoNotOptimizeAway(s);
        DoNotOptimizeAway(angle);
        DoNotOptimizeAway(result);
        });

    Benchmark("Mat4d::Inverse", []() {
        Mat4d inv = Mat4d().Inverse();
        DoNotOptimizeAway(inv);
        });

    Benchmark("Quaterniond::FromEuler & RotateVector", []() {
        double angle = 0.5;
        Vec3d v(1.0, 0.0, 0.0);
        Quaterniond q = Quaterniond::FromEuler(angle, angle, angle);
        Vec3d result = q.RotateVector(v);
        DoNotOptimizeAway(angle);
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(result);
        });

    Benchmark("Quaterniond::Slerp", []() {
        Quaterniond q1 = Quaterniond::Identity();
        double angle = 1.5;
        double t = 0.5;
        Quaterniond q2 = Quaterniond::FromEuler(0.0, angle, 0.0);
        Quaterniond result = Slerp(q1, q2, t);
        DoNotOptimizeAway(q1);
        DoNotOptimizeAway(angle);
        DoNotOptimizeAway(t);
        DoNotOptimizeAway(result);
        });
}

void BenchmarkSIMD() {
    std::println("Benchmark with SIMD optimizations:");
    std::println("{:-<122}", "");
    std::println("{:<4} | {:<40} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12}", "No.", "Name", "op/s", "ns/op", "err%", "cyc/op", "total (ms)");
    std::println("{:-<122}", "");

    Benchmark("VecSIMD<float, 4>::Normalize", []() {
        VecSIMD<float, 4> v(1.5f, 2.5f, 3.5f, 4.5f);
        VecSIMD<float, 4> res = v.Normalized();
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(res);
        });
    Benchmark("VecSIMD<float, 8>::Dot", []() {
        VecSIMD<float, 8> a(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f);
        VecSIMD<float, 8> b(8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f);
        float d = Dot(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(d);
        });
    Benchmark("VecSIMD<double, 2>::Normalize", []() {
        VecSIMD<double, 2> a(1.0, 2.0);
        VecSIMD<double, 2> b = a.Normalized();
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        });
    Benchmark("VecSIMD<double, 4>::Dot", []() {
        VecSIMD<double, 4> a(1.0, 2.0, 3.0, 4.0);
        VecSIMD<double, 4> b(5.0, 6.0, 7.0, 8.0);
        double d = Dot(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(d);
		});
}

int main() {
    BenchmarkNoSimd();
    BenchmarkSIMD();
    return 0;
}
