#pragma once

inline Vec3::Vec3()
    : x(0.0f), y(0.0f), z(0.0f)
{
}

inline Vec3::Vec3(float x, float y, float z)
    : x(x), y(y), z(z)
{
}

inline float Vec3::magnitude() const
{
    return std::sqrt(x * x + y * y + z * z);
}

inline Vec3 Vec3::normalized() const
{
    float mag = magnitude();

    if (mag == 0.0f)
        return Vec3(0.0f, 0.0f, 0.0f);

    return Vec3(
        x / mag,
        y / mag,
        z / mag
    );
}

inline float Vec3::Dot(const Vec3& first, const Vec3& second)
{
    return first.x * second.x
        + first.y * second.y
        + first.z * second.z;
}

inline float Vec3::sqrMagnitude() const
{
    return x * x + y * y + z * z;
}