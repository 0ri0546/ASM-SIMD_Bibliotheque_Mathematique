#include "Benchmark.h"

#include "Headers/Mat.h"
#include "Headers/Quaternion.h"
#include "Headers/Vec.h"

void BenchmarkNoSimd() {
    Benchmark("Vec3f::Normalize", []() {
        Vec3f v(1.5f, 2.5f, 3.5f);
        Vec3f res = v.Normalized();
        });

    Benchmark("Vec3f::DotAndCross", []() {
        Vec3f a(1.0f, 2.0f, 3.0f);
        Vec3f b(4.0f, 5.0f, 6.0f);
        float d = Dot(a, b);
        Vec3f c = Cross(a, b);
        });

    Benchmark("Mat4f::Translate & MultiplyPoint", []() {
        Vec3f point(1.0f, 2.0f, 3.0f);
        Mat4f translation = Mat4f::Translate(Vec3f(10.0f, 20.0f, 30.0f));
        Vec3f result = translation.MultiplyPoint(point);
        });

    Benchmark("Mat4f::TRS & Multiplication", []() {
        Mat4f m1 = Mat4f::TRS(Vec3f(1.f, 2.f, 3.f), Quaternionf::Identity(), Vec3f(2.f, 2.f, 2.f));
        Mat4f m2 = Mat4f::RotationX(0.5f);
        Mat4f result = m1 * m2;
        });

    Benchmark("Mat4f::Inverse", []() {
        Mat4f m = Mat4f::TRS(Vec3f(5.f, -2.f, 3.f), Quaternionf::FromEuler(0.1f, 0.2f, 0.3f), Vec3f(1.5f, 1.5f, 1.5f));
        Mat4f inv = m.Inverse();
        });

    Benchmark("Quaternionf::FromEuler & RotateVector", []() {
        Quaternionf q = Quaternionf::FromEuler(0.5f, 0.5f, 0.5f);
        Vec3f v(1.0f, 0.0f, 0.0f);
        Vec3f result = q.RotateVector(v);
        });

    Benchmark("Quaternionf::Slerp", []() {
        Quaternionf q1 = Quaternionf::Identity();
        Quaternionf q2 = Quaternionf::FromEuler(0.0f, 1.5f, 0.0f);
        Quaternionf result = Slerp(q1, q2, 0.5f);
        });

    Benchmark("Vec3d::Normalize", []() {
        Vec3d v(1.5, 2.5, 3.5);
        Vec3d res = v.Normalized();
        });

    Benchmark("Vec3d::DotAndCross", []() {
        Vec3d a(1.0, 2.0, 3.0);
        Vec3d b(4.0, 5.0, 6.0);
        double d = Dot(a, b);
        Vec3d c = Cross(a, b);
        });

    Benchmark("Mat4d::Translate & MultiplyPoint", []() {
        Vec3d point(1.0, 2.0, 3.0);
        Mat4d translation = Mat4d::Translate(Vec3d(10.0, 20.0, 30.0));
        Vec3d result = translation.MultiplyPoint(point);
        });

    Benchmark("Mat4d::TRS & Multiplication", []() {
        Mat4d m1 = Mat4d::TRS(Vec3d(1.0, 2.0, 3.0), Quaterniond::Identity(), Vec3d(2.0, 2.0, 2.0));
        Mat4d m2 = Mat4d::RotationX(0.5);
        Mat4d result = m1 * m2;
        });

    Benchmark("Mat4d::Inverse", []() {
        Mat4d m = Mat4d::TRS(Vec3d(5.0, -2.0, 3.0), Quaterniond::FromEuler(0.1, 0.2, 0.3), Vec3d(1.5, 1.5, 1.5));
        Mat4d inv = m.Inverse();
        });

    Benchmark("Quaterniond::FromEuler & RotateVector", []() {
        Quaterniond q = Quaterniond::FromEuler(0.5, 0.5, 0.5);
        Vec3d v(1.0, 0.0, 0.0);
        Vec3d result = q.RotateVector(v);
        });

    Benchmark("Quaterniond::Slerp", []() {
        Quaterniond q1 = Quaterniond::Identity();
        Quaterniond q2 = Quaterniond::FromEuler(0.0, 1.5, 0.0);
        Quaterniond result = Slerp(q1, q2, 0.5);
        });
}

int main() {
    std::println("Benchmark without SIMD optimizations:");
    std::println("-------------------------------------");
    std::println("{:<4} {:<40} {:>20} {:>20}", "No.", "Name", "Total (ms)", "Per iter (ns)\n");
    BenchmarkNoSimd();
	// TODO: Implement SIMD lib/optimizations and benchmark them
	// std::println("Benchmark with SIMD optimizations:");
    // std::println("-------------------------------------");
    // BenchmarkSimd();
    // std::println("-------------------------------------");
    return 0;
}
