#include <iostream>

#include "Headers/Config/SimdConfig.h"
#include "Headers/Mat.h"
#include "Headers/Quaternion.h"
#include "Headers/Vec.h"

int main()
{
    if(!Simd::Has(Simd::Feature::sse))
    {
        std::cerr << "SSE not supported\n";
        return -1;
    }

    Vec3f point(1.0f, 2.0f, 3.0f);

    Mat4f translation = Mat4f::Translate(Vec3f(10.0f, 20.0f, 30.0f));

    Vec3f result = translation.MultiplyPoint(point);

    std::cout
        << result[0] << ' '
        << result[1] << ' '
        << result[2] << '\n';
}