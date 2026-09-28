#pragma once

#include <cmath>

class Vec3
{
public:
    float x;
    float y;
    float z;

public:
    Vec3();
    Vec3(float x, float y, float z);

    float magnitude() const;
    float sqrMagnitude() const;
    Vec3 normalized() const;

    static float Dot(const Vec3& first, const Vec3& second);
};

#include "Vec3Reference.inl"