#include <print>

#include "Headers/Config/SimdConfig.h"
#include "Headers/Mat.h"
#include "Headers/Quaternion.h"
#include "Headers/Vec.h"

int main()
{
    if(!Simd::Has(Simd::Feature::sse))
    {
        std::println("SSE not supported");
        return -1;
    }

    Vec3f point(1.0f, 2.0f, 3.0f);

    Mat4f translation = Mat4f::Translate(Vec3f(10.0f, 20.0f, 30.0f));

    Vec3f result = translation.MultiplyPoint(point);

    std::println("{}", result);
}
