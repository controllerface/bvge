inline float4 matrix_transform(float16 matrix, float4 vector)
{
    float4 result;
    result.x = fma(matrix.s0, vector.x, fma(matrix.s4, vector.y, fma(matrix.s8, vector.z, matrix.sC * vector.w)));
    result.y = fma(matrix.s1, vector.x, fma(matrix.s5, vector.y, fma(matrix.s9, vector.z, matrix.sD * vector.w)));
    result.z = fma(matrix.s2, vector.x, fma(matrix.s6, vector.y, fma(matrix.sA, vector.z, matrix.sE * vector.w)));
    result.w = fma(matrix.s3, vector.x, fma(matrix.s7, vector.y, fma(matrix.sB, vector.z, matrix.sF * vector.w)));
    return result;
}