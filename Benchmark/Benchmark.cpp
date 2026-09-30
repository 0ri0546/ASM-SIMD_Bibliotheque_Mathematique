#include "Benchmark.h"

#include "Headers/Mat.h"
#include "Headers/Quaternion.h"
#include "Headers/Vec.h"

void BenchmarkNoSimd() {
    Benchmark("Vec3f::Normalize", []() {
        Vec3f v(1.5f, 2.5f, 3.5f);
        Vec3f res = v.Normalized();
        DoNotOptimize(v);
        DoNotOptimize(res);
        });

    Benchmark("Vec3f::DotAndCross", []() {
        Vec3f a(1.0f, 2.0f, 3.0f);
        Vec3f b(4.0f, 5.0f, 6.0f);
        float d = Dot(a, b);
        Vec3f c = Cross(a, b);
        DoNotOptimize(a);
        DoNotOptimize(b);
        DoNotOptimize(d);
        DoNotOptimize(c);
        });

    Benchmark("Mat4f::Translate & MultiplyPoint", []() {
        Vec3f point(1.0f, 2.0f, 3.0f);
        Vec3f offset(10.0f, 20.0f, 30.0f);
        Mat4f translation = Mat4f::Translate(offset);
        Vec3f result = translation.MultiplyPoint(point);
        DoNotOptimize(point);
        DoNotOptimize(offset);
        DoNotOptimize(translation);
        DoNotOptimize(result);
        });

    Benchmark("Mat4f::TRS & Multiplication", []() {
        Vec3f t(1.f, 2.f, 3.f);
        Quaternionf r = Quaternionf::Identity();
        Vec3f s(2.f, 2.f, 2.f);
        Mat4f m1 = Mat4f::TRS(t, r, s);
        Mat4f m2 = Mat4f::RotationX(0.5f);
        Mat4f result = m1 * m2;
        DoNotOptimize(t);
        DoNotOptimize(r);
        DoNotOptimize(s);
        DoNotOptimize(m1);
        DoNotOptimize(m2);
        DoNotOptimize(result);
        });

    Benchmark("Mat4f::Inverse", []() {
        Vec3f t(5.f, -2.f, 3.f);
        Quaternionf r = Quaternionf::FromEuler(0.1f, 0.2f, 0.3f);
        Vec3f s(1.5f, 1.5f, 1.5f);
        Mat4f m = Mat4f::TRS(t, r, s);
        Mat4f inv = m.Inverse();
        DoNotOptimize(t);
        DoNotOptimize(r);
        DoNotOptimize(s);
        DoNotOptimize(m);
        DoNotOptimize(inv);
        });

    Benchmark("Quaternionf::FromEuler & RotateVector", []() {
        float angle = 0.5f;
        Vec3f v(1.0f, 0.0f, 0.0f);
        Quaternionf q = Quaternionf::FromEuler(angle, angle, angle);
        Vec3f result = q.RotateVector(v);
        DoNotOptimize(q);
        DoNotOptimize(angle);
        DoNotOptimize(v);
        DoNotOptimize(result);
        });

    Benchmark("Quaternionf::Slerp", []() {
        Quaternionf q1 = Quaternionf::Identity();
        float angle = 1.5f;
        float t = 0.5f;
        Quaternionf q2 = Quaternionf::FromEuler(0.0f, angle, 0.0f);
        Quaternionf result = Slerp(q1, q2, t);
        DoNotOptimize(q1);
        DoNotOptimize(angle);
        DoNotOptimize(t);
        DoNotOptimize(q2);
        DoNotOptimize(result);
        });

    Benchmark("Vec3d::Normalize", []() {
        Vec3d v(1.5, 2.5, 3.5);
        Vec3d res = v.Normalized();
        DoNotOptimize(v);
        DoNotOptimize(res);
        });

    Benchmark("Vec3d::DotAndCross", []() {
        Vec3d a(1.0, 2.0, 3.0);
        Vec3d b(4.0, 5.0, 6.0);
        double d = Dot(a, b);
        Vec3d c = Cross(a, b);
        DoNotOptimize(a);
        DoNotOptimize(b);
        DoNotOptimize(d);
        DoNotOptimize(c);
        });

    Benchmark("Mat4d::Translate & MultiplyPoint", []() {
        Vec3d point(1.0, 2.0, 3.0);
        Vec3d offset(10.0, 20.0, 30.0);
        Mat4d translation = Mat4d::Translate(offset);
        Vec3d result = translation.MultiplyPoint(point);
        DoNotOptimize(point);
        DoNotOptimize(offset);
        DoNotOptimize(translation);
        DoNotOptimize(result);
        });

    Benchmark("Mat4d::TRS & Multiplication", []() {
        Vec3d t(1.0, 2.0, 3.0);
        Quaterniond r = Quaterniond::Identity();
        Vec3d s(2.0, 2.0, 2.0);
        Mat4d m1 = Mat4d::TRS(t, r, s);
        Mat4d m2 = Mat4d::RotationX(0.5);
        Mat4d result = m1 * m2;
        DoNotOptimize(t);
        DoNotOptimize(r);
        DoNotOptimize(s);
        DoNotOptimize(m1);
        DoNotOptimize(m2);
        DoNotOptimize(result);
        });

    Benchmark("Mat4d::Inverse", []() {
        Vec3d t(5.0, -2.0, 3.0);
        Quaterniond r = Quaterniond::FromEuler(0.1, 0.2, 0.3);
        Vec3d s(1.5, 1.5, 1.5);
        Mat4d m = Mat4d::TRS(t, r, s);
        Mat4d inv = m.Inverse();
        DoNotOptimize(t);
        DoNotOptimize(r);
        DoNotOptimize(s);
        DoNotOptimize(m);
        DoNotOptimize(inv);
        });

    Benchmark("Quaterniond::FromEuler & RotateVector", []() {
        double angle = 0.5;
        Vec3d v(1.0, 0.0, 0.0);
        Quaterniond q = Quaterniond::FromEuler(angle, angle, angle);
        Vec3d result = q.RotateVector(v);
        DoNotOptimize(angle);
        DoNotOptimize(v);
        DoNotOptimize(q);
        DoNotOptimize(result);
        });

    Benchmark("Quaterniond::Slerp", []() {
        Quaterniond q1 = Quaterniond::Identity();
        double angle = 1.5;
        double t = 0.5;
        Quaterniond q2 = Quaterniond::FromEuler(0.0, angle, 0.0);
        Quaterniond result = Slerp(q1, q2, t);
        DoNotOptimize(q1);
        DoNotOptimize(angle);
        DoNotOptimize(t);
        DoNotOptimize(q2);
        DoNotOptimize(result);
        });
}

int main() {
    std::println("Benchmark without SIMD optimizations:");
    std::println("--------------------------------------------------------------------------------------------------------------------------");
    std::println("{:<4} | {:<40} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12}", "No.", "Name", "op/s", "ns/op", "err%", "cyc/op", "total (ms)");
    std::println("--------------------------------------------------------------------------------------------------------------------------");
    BenchmarkNoSimd();
    // TODO: Implement SIMD lib/optimizations and benchmark them
    // std::println("Benchmark with SIMD optimizations:");
    // std::println("-------------------------------------");
    // BenchmarkSimd();
    // std::println("-------------------------------------");
    return 0;
}
