inline float4 matrix_transform(float16 matrix, float4 vector)
{
    float4 result;
    result.x = mad(matrix.s0, vector.x, mad(matrix.s4, vector.y, mad(matrix.s8, vector.z, matrix.sC * vector.w)));
    result.y = mad(matrix.s1, vector.x, mad(matrix.s5, vector.y, mad(matrix.s9, vector.z, matrix.sD * vector.w)));
    result.z = mad(matrix.s2, vector.x, mad(matrix.s6, vector.y, mad(matrix.sA, vector.z, matrix.sE * vector.w)));
    result.w = mad(matrix.s3, vector.x, mad(matrix.s7, vector.y, mad(matrix.sB, vector.z, matrix.sF * vector.w)));
    return result;
}