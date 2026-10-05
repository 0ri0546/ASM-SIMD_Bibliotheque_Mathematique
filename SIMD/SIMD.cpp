#include <print>

#include "VecSIMD.h"
#include "MatSIMD.h"
#include "QuaternionSIMD.h"

int main()
{
    VecSIMD<float, 4> a(1.0f, 2.0f, 3.0f, 4.0f);
    VecSIMD<float, 4> b(5.0f, 6.0f, 7.0f, 8.0f);

	std::println("a = {:.3f}", a);
	std::println("b = {:.3f}", b);

    std::println("a + b = {:.3f}", a + b);
    std::println("a - b = {:.3f}", a - b);
    std::println("a * b = {:.3f}", a * b);
    std::println("a / b = {:.3f}", a / b);
    std::println("a * 2.f = {:.3f}", a * 2.f);
    std::println("-a = {:.3f}", -a);
    std::println("Dot(a, b) = {:.3f}", Dot(a, b));

    std::println("a.LengthSquared() = {:.3f}", a.LengthSquared());
    std::println("a.Length() = {:.3f}", a.Length());
    std::println("a.Normalized() = {:.3f}", a.Normalized());
    VecSIMD<float, 8> c(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f);
    VecSIMD<float, 8> d(8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f);

    std::println("c + d = {:.3f}", c + d);
    std::println("c - d = {:.3f}", c - d);
    std::println("c * d = {:.3f}", c * d);
    std::println("c / d = {:.3f}", c / d);

    std::println("c * 2.f = {:.3f}", c * 2.f);
    std::println("Dot(c, d) = {:.3f}", Dot(c, d));

    VecSIMD<double, 2> e(1.0, 2.0);
    VecSIMD<double, 2> f(3.0, 4.0);

    std::println("e + f = {:.3f}", e + f);
    std::println("e - f = {:.3f}", e - f);
    std::println("e * f = {:.3f}", e * f);
    std::println("e / f = {:.3f}", e / f);
    std::println("Dot(e, f) = {:.3f}", Dot(e, f));
    VecSIMD<double, 4> g(1.0, 2.0, 3.0, 4.0);
    VecSIMD<double, 4> h(5.0, 6.0, 7.0, 8.0);

    std::println("g + h = {:.3f}", g + h);
    std::println("g - h = {:.3f}", g - h);
    std::println("g * h = {:.3f}", g * h);
    std::println("g / h = {:.3f}", g / h);
    std::println("Dot(g, h) = {:.3f}", Dot(g, h));
    std::println("");
    std::println("--------------------------------QUATERNION----------------------------------");
    std::println("");

    QuaternionSIMDf i(1.0f, 2.0f, 3.0f, 4.0f);
    QuaternionSIMDf j(5.0f, 6.0f, 7.0f, 8.0f);
    std::println("g + h = {:.3f}", i + j);
    std::println("g - h = {:.3f}", i - j);
    std::println("g * h = {:.3f}", i * j);
    std::println("g / h = {:.3f}", i / 2);
    std::println("identity = {}", i.Identity());
    std::println("FromEuler = {}", i.FromEuler(1.0, 2.0, 3.0));
    VecSIMD<float, 3> vec3(1.0f, 2.0f, 3.0f);
    VecSIMD<float, 3> vec3b(4.0f, 3.0f, 12.0f);
    std::println("FromAxisAngle = {}", i.FromAxisAngle(vec3, 4.0f));
    std::println("FromToRotation = {}", i.FromToRotation(vec3, vec3b));
    std::println("LookRotation = {}", i.LookRotation(vec3, vec3b));
    std::println("");
    std::println("--------------------------------MATRIX----------------------------------");
    std::println("");

    MatSIMD<double, 8, 3> mat_a = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
    mat_a *= 2.0f;
    mat_a /= 4.0f;
    std::println("{}", mat_a);

    MatSIMD<double, 4, 3> mat_b = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    mat_b *= 2.0f;
    mat_b /= 4.0f;
    std::println("{}", mat_b);

    MatSIMD<double, 3, 3> mat_c = { 1,2,3,4,5,6,7,8,9 };
    mat_c *= 2.0f;
    mat_c /= 4.0f;
    std::println("{}", mat_c);

    MatSIMD<double, 8, 3, ColumnMajor> mat_d = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
    mat_d *= 2.0f;
    mat_d /= 4.0f;
    std::println("{}", mat_d);

    MatSIMD<double, 4, 3, ColumnMajor> mat_e = { 1,2,3,4,5,6,7,8,9,10,11,12 };
    mat_e *= 2.0f;
    mat_e /= 4.0f;
    std::println("{}", mat_e);

    MatSIMD<double, 3, 3, ColumnMajor> mat_f = { 1,2,3,4,5,6,7,8,9 };
    mat_f *= 2.0f;
    mat_f /= 4.0f;
    std::println("{}", mat_f);

	auto mat_g = MatCastLayout<double, 3, 3, ColumnMajor, RowMajor>(mat_f);
    std::println("{}", mat_g);

    return 0;
}
