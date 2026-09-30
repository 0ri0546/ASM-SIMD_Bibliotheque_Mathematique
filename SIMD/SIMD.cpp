#include <print>

#include <iostream>
#include "VecSIMD.h"

int main()
{
    Vec4d point(1.0f, 2.0f, 3.0f, 4.0f);
    Vec4d point2(1.0f, 2.0f, 3.0f, 4.0f);
    auto result = point + point2;
    std::println("{}", result[0]);
    std::println("{}", result.get_m128<0>());
    std::println("{}", result.get_m128<1>());
    std::println("{}", result.get_m256<0>());
}