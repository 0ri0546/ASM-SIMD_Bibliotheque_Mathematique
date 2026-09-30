#include <print>

#include <iostream>
#include "VecSIMD.h"

int main()
{
    Vec4f point(1.0f, 2.0f, 3.0f, 4.0f);
    Vec4f point2(1.0f, 2.0f, 3.0f, 4.0f);

    std::println("{}", point[2]);

    Vec4f result = point + point2;
    std::println("{}", result.to_m128());

}