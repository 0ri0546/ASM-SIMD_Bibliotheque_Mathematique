#include <print>

#include "VecSIMD.h"

int main()
{
    VecSIMD<float, 4> a(1.f, 2.f, 3.f, 4.f);
    VecSIMD<float, 4> b(5.f, 6.f, 7.f, 8.f);

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
    return 0;
}
