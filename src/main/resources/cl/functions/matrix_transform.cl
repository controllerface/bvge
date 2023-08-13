inline float4 matrix_transform(float16 matrix, float4 vector)
{
    float4 result;
    result.x = matrix.s0 * vector.x + matrix.s4 * vector.y + matrix.s8 * vector.z + matrix.sC * vector.w;
    result.y = matrix.s1 * vector.x + matrix.s5 * vector.y + matrix.s9 * vector.z + matrix.sD * vector.w;
    result.z = matrix.s2 * vector.x + matrix.s6 * vector.y + matrix.sA * vector.z + matrix.sE * vector.w;
    result.w = matrix.s3 * vector.x + matrix.s7 * vector.y + matrix.sB * vector.z + matrix.sF * vector.w;
    return result;
}