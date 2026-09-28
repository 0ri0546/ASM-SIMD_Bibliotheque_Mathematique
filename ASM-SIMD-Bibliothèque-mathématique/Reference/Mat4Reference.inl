#pragma once

inline Mat4::Mat4()
{
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            data[i][j] = 0.0f;
        }
    }
}

inline Mat4::Mat4(
    float m00, float m01, float m02, float m03,
    float m10, float m11, float m12, float m13,
    float m20, float m21, float m22, float m23,
    float m30, float m31, float m32, float m33)
{
    data[0][0] = m00;
    data[0][1] = m01;
    data[0][2] = m02;
    data[0][3] = m03;

    data[1][0] = m10;
    data[1][1] = m11;
    data[1][2] = m12;
    data[1][3] = m13;

    data[2][0] = m20;
    data[2][1] = m21;
    data[2][2] = m22;
    data[2][3] = m23;

    data[3][0] = m30;
    data[3][1] = m31;
    data[3][2] = m32;
    data[3][3] = m33;
}

inline Mat4 Mat4::Identity()
{
    return Mat4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );
}

inline Mat4 Mat4::Translate(const Vec3& position)
{
    Mat4 result = Identity();

    result.data[0][3] = position.x;
    result.data[1][3] = position.y;
    result.data[2][3] = position.z;

    return result;
}

inline Mat4 Mat4::Scale(const Vec3& scale)
{
    Mat4 result = Identity();

    result.data[0][0] = scale.x;
    result.data[1][1] = scale.y;
    result.data[2][2] = scale.z;

    return result;
}

inline Mat4 Mat4::operator*(const Mat4& other) const
{
    Mat4 result;

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            result.data[i][j] =
                data[i][0] * other.data[0][j] +
                data[i][1] * other.data[1][j] +
                data[i][2] * other.data[2][j] +
                data[i][3] * other.data[3][j];
        }
    }

    return result;
}

inline Vec3 Mat4::MultiplyPoint3x4(const Vec3& point) const
{
    float x =
        data[0][0] * point.x +
        data[0][1] * point.y +
        data[0][2] * point.z +
        data[0][3];

    float y =
        data[1][0] * point.x +
        data[1][1] * point.y +
        data[1][2] * point.z +
        data[1][3];

    float z =
        data[2][0] * point.x +
        data[2][1] * point.y +
        data[2][2] * point.z +
        data[2][3];

    return Vec3(x, y, z);
}