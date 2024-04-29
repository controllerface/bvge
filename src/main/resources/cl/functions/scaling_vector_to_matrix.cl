inline float16 scaling_vector_to_matrix(float4 vector)
{
    float16 matrix;
    matrix.s0 = vector.x;
    matrix.s1 = 0;
    matrix.s2 = 0;
    matrix.s3 = 0;
    matrix.s4 = 0;
    matrix.s5 = vector.y;
    matrix.s6 = 0;
    matrix.s7 = 0;
    matrix.s8 = 0;
    matrix.s9 = 0;
    matrix.sA = vector.z;
    matrix.sB = 0;
    matrix.sC = 0;
    matrix.sD = 0;
    matrix.sE = 0;
    matrix.sF = vector.w;
    return matrix;
}
