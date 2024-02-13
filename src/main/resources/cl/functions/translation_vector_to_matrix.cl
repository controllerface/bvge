inline float16 translation_vector_to_matrix(float4 vector)
{
    float16 matrix;
    matrix.s0 = 1;
    matrix.s1 = 0;
    matrix.s2 = 0;
    matrix.s3 = 0;
    matrix.s4 = 0;
    matrix.s5 = 1;
    matrix.s6 = 0;
    matrix.s7 = 0;
    matrix.s8 = 0;
    matrix.s9 = 0;
    matrix.sA = 1;
    matrix.sB = 0;
    matrix.sC = vector.x;
    matrix.sD = vector.y;
    matrix.sE = vector.z;
    matrix.sF = vector.w;
    return matrix;
}