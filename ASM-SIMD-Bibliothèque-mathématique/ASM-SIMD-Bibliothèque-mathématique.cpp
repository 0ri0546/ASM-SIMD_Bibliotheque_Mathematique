#include <iostream>

#include "Reference/Vec3Reference.h"
#include "Reference/Mat4Reference.h"

int main()
{
    Vec3 point(1.0f, 2.0f, 3.0f);

    Mat4 translation = Mat4::Translate(Vec3(10.0f, 20.0f, 30.0f));

    Vec3 result = translation.MultiplyPoint3x4(point);

    std::cout
        << result.x << ' '
        << result.y << ' '
        << result.z << '\n';
}