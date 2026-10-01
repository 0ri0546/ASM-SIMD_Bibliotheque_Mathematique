#include <print>

#include "VecSIMD.h"
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

    Quaternionf i(1.0f, 2.0f, 3.0f, 4.0f);
    Quaternionf j(5.0f, 6.0f, 7.0f, 8.0f);
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

    return 0;
}
