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

    Benchmark("Vec4f::Normalize", []() {
        Vec4f v(1.5f, 2.5f, 3.5f, 4.5f);
        Vec4f res = v.Normalized();
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4f::Dot", []() {
        Vec4f a(1.0f, 2.0f, 3.0f, 4.0f);
        Vec4f b(4.0f, 5.0f, 6.0f, 7.0f);
        float d = Dot(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(d);
        });
    Benchmark("Vec2d::Normalize", []() {
        Vec2d v(1.5, 2.5);
        Vec2d res = v.Normalized();
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec2d::Dot", []() {
        Vec2d a(1.0, 2.0);
        Vec2d b(4.0, 5.0);
        double d = Dot(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(d);
        });
    Benchmark("Vec4d::Normalize", []() {
        Vec4d v(1.5, 2.5, 3.5, 4.5);
        Vec4d res = v.Normalized();
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4d::Dot", []() {
        Vec4d a(1.0, 2.0, 3.0, 4.0);
        Vec4d b(4.0, 5.0, 6.0, 7.0);
        double d = Dot(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(d);
        });
    Benchmark("Vec8f::Normalize", []() {
        Vec<float, 8> v(1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f);
        Vec<float, 8> res = v.Normalized();
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec8f::Dot", []() {
        Vec<float, 8> a(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
        Vec<float, 8> b(4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f);
        float d = Dot(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(d);
        });
    // TODO: Implement with SIMD:
    Benchmark("Vec3f::Cross", []() {
        Vec3f a(1.0f, 2.0f, 3.0f);
        Vec3f b(4.0f, 5.0f, 6.0f);
        Vec3f res = Cross(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec2f::Perpendicular", []() {
        Vec2f v(1.5f, 2.5f);
        Vec2f res = Perpendicular(v);
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4f::Distance", []() {
        Vec4f a(1.0f, 2.0f, 3.0f, 4.0f);
        Vec4f b(4.0f, 5.0f, 6.0f, 7.0f);
        float d = Distance(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(d);
        });

    Benchmark("Vec4f::Lerp", []() {
        Vec4f a(1.0f, 2.0f, 3.0f, 4.0f);
        Vec4f b(4.0f, 5.0f, 6.0f, 7.0f);
        float t = 0.5f;
        Vec4f res = Lerp(a, b, t);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(t);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4f::Scale", []() {
        Vec4f a(1.0f, 2.0f, 3.0f, 4.0f);
        Vec4f b(4.0f, 5.0f, 6.0f, 7.0f);
        Vec4f res = Scale(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4f::Min", []() {
        Vec4f a(1.0f, 5.0f, 3.0f, 7.0f);
        Vec4f b(4.0f, 2.0f, 6.0f, 4.0f);
        Vec4f res = Min(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4f::Max", []() {
        Vec4f a(1.0f, 5.0f, 3.0f, 7.0f);
        Vec4f b(4.0f, 2.0f, 6.0f, 4.0f);
        Vec4f res = Max(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4f::MoveTowards", []() {
        Vec4f current(1.0f, 2.0f, 3.0f, 4.0f);
        Vec4f target(10.0f, 11.0f, 12.0f, 13.0f);
        float maxDelta = 2.5f;
        Vec4f res = MoveTowards(current, target, maxDelta);
        DoNotOptimizeAway(current);
        DoNotOptimizeAway(target);
        DoNotOptimizeAway(maxDelta);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4f::Reflect", []() {
        Vec4f v(1.0f, -2.0f, 3.0f, -4.0f);
        Vec4f n(0.0f, 1.0f, 0.0f, 0.0f);
        Vec4f res = Reflect(v, n);
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(n);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4f::Angle", []() {
        Vec4f a(1.0f, 2.0f, 3.0f, 4.0f);
        Vec4f b(4.0f, 5.0f, 6.0f, 7.0f);
        float ang = Angle(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(ang);
        });

    Benchmark("Vec3d::Cross", []() {
        Vec3d a(1.0, 2.0, 3.0);
        Vec3d b(4.0, 5.0, 6.0);
        Vec3d res = Cross(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec2d::Perpendicular", []() {
        Vec2d v(1.5, 2.5);
        Vec2d res = Perpendicular(v);
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4d::Distance", []() {
        Vec4d a(1.0, 2.0, 3.0, 4.0);
        Vec4d b(4.0, 5.0, 6.0, 7.0);
        double d = Distance(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(d);
        });

    Benchmark("Vec4d::Lerp", []() {
        Vec4d a(1.0, 2.0, 3.0, 4.0);
        Vec4d b(4.0, 5.0, 6.0, 7.0);
        double t = 0.5;
        Vec4d res = Lerp(a, b, t);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(t);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4d::Scale", []() {
        Vec4d a(1.0, 2.0, 3.0, 4.0);
        Vec4d b(4.0, 5.0, 6.0, 7.0);
        Vec4d res = Scale(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4d::Min", []() {
        Vec4d a(1.0, 5.0, 3.0, 7.0);
        Vec4d b(4.0, 2.0, 6.0, 4.0);
        Vec4d res = Min(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4d::Max", []() {
        Vec4d a(1.0, 5.0, 3.0, 7.0);
        Vec4d b(4.0, 2.0, 6.0, 4.0);
        Vec4d res = Max(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4d::MoveTowards", []() {
        Vec4d current(1.0, 2.0, 3.0, 4.0);
        Vec4d target(10.0, 11.0, 12.0, 13.0);
        double maxDelta = 2.5;
        Vec4d res = MoveTowards(current, target, maxDelta);
        DoNotOptimizeAway(current);
        DoNotOptimizeAway(target);
        DoNotOptimizeAway(maxDelta);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4d::Reflect", []() {
        Vec4d v(1.0, -2.0, 3.0, -4.0);
        Vec4d n(0.0, 1.0, 0.0, 0.0);
        Vec4d res = Reflect(v, n);
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(n);
        DoNotOptimizeAway(res);
        });

    Benchmark("Vec4d::Angle", []() {
        Vec4d a(1.0, 2.0, 3.0, 4.0);
        Vec4d b(4.0, 5.0, 6.0, 7.0);
        double ang = Angle(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(ang);
        });

    Benchmark("Mat4f::Translate", []() {
        Vec3f offset(10.0f, 20.0f, 30.0f);
        Mat4f translation = Mat4f::Translate(offset);
        DoNotOptimizeAway(offset);
        DoNotOptimizeAway(translation);
        });

    Benchmark("Mat4f::MultiplyPoint", []() {
        static const Mat4f translation = Mat4f::Translate(Vec3f(10.0f, 20.0f, 30.0f));
        Vec3f point(1.0f, 2.0f, 3.0f);
        Vec3f result = translation.MultiplyPoint(point);
        DoNotOptimizeAway(translation);
        DoNotOptimizeAway(point);
        DoNotOptimizeAway(result);
        });

    Benchmark("Mat4f::TRS", []() {
        Vec3f t(1.f, 2.f, 3.f);
        Quaternionf r = Quaternionf::Identity();
        Vec3f s(2.f, 2.f, 2.f);
        Mat4f result = Mat4f::TRS(t, r, s);
        DoNotOptimizeAway(t);
        DoNotOptimizeAway(r);
        DoNotOptimizeAway(s);
        DoNotOptimizeAway(result);
        });

    Benchmark("Mat4f::RotationX", []() {
        float angle = 0.5f;
        Mat4f result = Mat4f::RotationX(angle);
        DoNotOptimizeAway(angle);
        DoNotOptimizeAway(result);
        });

    Benchmark("Mat4f::Multiplication", []() {
        static const Mat4f m1 = Mat4f::TRS(Vec3f(1.f, 2.f, 3.f), Quaternionf::Identity(), Vec3f(2.f, 2.f, 2.f));
        static const Mat4f m2 = Mat4f::RotationX(0.5f);
        Mat4f result = m1 * m2;
        DoNotOptimizeAway(m1);
        DoNotOptimizeAway(m2);
        DoNotOptimizeAway(result);
        });

    Benchmark("Mat4f::Inverse", []() {
        Mat4f inv = Mat4f().Inverse();
        DoNotOptimizeAway(inv);
        });

    Benchmark("Quaternionf::FromEuler", []() {
        float angle = 0.5f;
        Quaternionf q = Quaternionf::FromEuler(angle, angle, angle);
        DoNotOptimizeAway(angle);
        DoNotOptimizeAway(q);
        });

    Benchmark("Quaternionf::RotateVector", []() {
        static const Quaternionf q = Quaternionf::FromEuler(0.5f, 0.5f, 0.5f);
        Vec3f v(1.0f, 0.0f, 0.0f);
        Vec3f result = q.RotateVector(v);
        DoNotOptimizeAway(q);
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(result);
        });

    Benchmark("Quaternionf::Slerp", []() {
        static const Quaternionf q1 = Quaternionf::Identity();
        static const Quaternionf q2 = Quaternionf::FromEuler(0.0f, 1.5f, 0.0f);
        float t = 0.5f;
        Quaternionf result = Slerp(q1, q2, t);
        DoNotOptimizeAway(q1);
        DoNotOptimizeAway(q2);
        DoNotOptimizeAway(t);
        DoNotOptimizeAway(result);
        });

    Benchmark("Mat4d::Translate", []() {
        Vec3d offset(10.0, 20.0, 30.0);
        Mat4d translation = Mat4d::Translate(offset);
        DoNotOptimizeAway(offset);
        DoNotOptimizeAway(translation);
        });

    Benchmark("Mat4d::MultiplyPoint", []() {
        static const Mat4d translation = Mat4d::Translate(Vec3d(10.0, 20.0, 30.0));
        Vec3d point(1.0, 2.0, 3.0);
        Vec3d result = translation.MultiplyPoint(point);
        DoNotOptimizeAway(translation);
        DoNotOptimizeAway(point);
        DoNotOptimizeAway(result);
        });

    Benchmark("Mat4d::TRS", []() {
        Vec3d t(1.0, 2.0, 3.0);
        Quaterniond r = Quaterniond::Identity();
        Vec3d s(2.0, 2.0, 2.0);
        Mat4d result = Mat4d::TRS(t, r, s);
        DoNotOptimizeAway(t);
        DoNotOptimizeAway(r);
        DoNotOptimizeAway(s);
        DoNotOptimizeAway(result);
        });

    Benchmark("Mat4d::RotationX", []() {
        double angle = 0.5;
        Mat4d result = Mat4d::RotationX(angle);
        DoNotOptimizeAway(angle);
        DoNotOptimizeAway(result);
        });

    Benchmark("Mat4d::Multiplication", []() {
        static const Mat4d m1 = Mat4d::TRS(Vec3d(1.0, 2.0, 3.0), Quaterniond::Identity(), Vec3d(2.0, 2.0, 2.0));
        static const Mat4d m2 = Mat4d::RotationX(0.5);
        Mat4d result = m1 * m2;
        DoNotOptimizeAway(m1);
        DoNotOptimizeAway(m2);
        DoNotOptimizeAway(result);
        });

    Benchmark("Mat4d::Inverse", []() {
        Mat4d inv = Mat4d().Inverse();
        DoNotOptimizeAway(inv);
        });

    Benchmark("Quaterniond::FromEuler", []() {
        double angle = 0.5;
        Quaterniond q = Quaterniond::FromEuler(angle, angle, angle);
        DoNotOptimizeAway(angle);
        DoNotOptimizeAway(q);
        });

    Benchmark("Quaterniond::RotateVector", []() {
        static const Quaterniond q = Quaterniond::FromEuler(0.5, 0.5, 0.5);
        Vec3d v(1.0, 0.0, 0.0);
        Vec3d result = q.RotateVector(v);
        DoNotOptimizeAway(q);
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(result);
        });

    Benchmark("Quaterniond::Slerp", []() {
        static const Quaterniond q1 = Quaterniond::Identity();
        static const Quaterniond q2 = Quaterniond::FromEuler(0.0, 1.5, 0.0);
        double t = 0.5;
        Quaterniond result = Slerp(q1, q2, t);
        DoNotOptimizeAway(q1);
        DoNotOptimizeAway(q2);
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
    Benchmark("VecSIMD<float, 4>::Dot", []() {
        VecSIMD<float, 4> v(1.5f, 2.5f, 3.5f, 4.5f);
        float res = Dot(v, v);
        DoNotOptimizeAway(v);
        DoNotOptimizeAway(res);
        });
    Benchmark("VecSIMD<double, 2>::Normalize", []() {
        VecSIMD<double, 2> a(1.0, 2.0);
        VecSIMD<double, 2> b = a.Normalized();
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        });
    Benchmark("VecSIMD<double, 2>::Dot", []() {
        VecSIMD<double, 2> a(1.0, 2.0);
        VecSIMD<double, 2> b(5.0, 6.0);
        double d = Dot(a, b);
        DoNotOptimizeAway(a);
        DoNotOptimizeAway(b);
        DoNotOptimizeAway(d);
        });
    Benchmark("VecSIMD<double, 4>::Normalize", []() {
        VecSIMD<double, 4> a(1.0, 2.0, 3.0, 4.0);
        VecSIMD<double, 4> b = a.Normalized();
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
    Benchmark("VecSIMD<float, 8>::Normalize", []() {
        VecSIMD<float, 8> v(1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f);
        VecSIMD<float, 8> res = v.Normalized();
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
}

int main() {
    BenchmarkNoSimd();
    BenchmarkSIMD();
    return 0;
}
