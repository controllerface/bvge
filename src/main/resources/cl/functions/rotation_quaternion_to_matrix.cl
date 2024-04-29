inline float16 rotation_quaternion_to_matrix(float4 quaternion)
{
    float16 matrix;
    float w2 = quaternion.w * quaternion.w;
    float x2 = quaternion.x * quaternion.x;
    float y2 = quaternion.y * quaternion.y;
    float z2 = quaternion.z * quaternion.z;
    float zw = quaternion.z * quaternion.w;
    float xy = quaternion.x * quaternion.y;
    float xz = quaternion.x * quaternion.z;
    float yw = quaternion.y * quaternion.w;
    float yz = quaternion.y * quaternion.z;
    float xw = quaternion.x * quaternion.w;
    matrix.s0 = w2 + x2 - z2 - y2;
    matrix.s1 = xy + zw + zw + xy;
    matrix.s2 = xz - yw + xz - yw;
    matrix.s3 = 0.0F;
    matrix.s4 = -zw + xy - zw + xy;
    matrix.s5 = y2 - z2 + w2 - x2;
    matrix.s6 = yz + yz + xw + xw;
    matrix.s7 = 0.0F;
    matrix.s8 = yw + xz + xz + yw;
    matrix.s9 = yz + yz - xw - xw;
    matrix.sA = z2 - y2 - x2 + w2;
    matrix.sB = 0.0F;
    matrix.sC = 0.0F;
    matrix.sD = 0.0F;
    matrix.sE = 0.0F;
    matrix.sF = 1.0F;
    return matrix;
}
