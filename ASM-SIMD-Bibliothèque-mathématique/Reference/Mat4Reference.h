#pragma once

#include <cmath>
#include "Vec3Reference.h"

class Mat4
{

public:
    Mat4();
    Mat4(float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33);


    float data[4][4];

    static Mat4 Identity();
    static Mat4 Translate(const Vec3& position);
    static Mat4 Scale(const Vec3& scale);

    Mat4 operator*(const Mat4& other) const;

    Vec3 MultiplyPoint3x4(const Vec3& point) const;
};

#include "Mat4Reference.inl"