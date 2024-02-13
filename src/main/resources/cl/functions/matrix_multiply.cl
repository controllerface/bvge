inline float16 matrix_mul(float16 matrixA, float16 matrixB) 
{
    float16 result;

    result.s0 = fma(matrixA.s0, matrixB.s0, fma(matrixA.s4, matrixB.s1, fma(matrixA.s8, matrixB.s2, matrixA.sC * matrixB.s3)));
    result.s1 = fma(matrixA.s1, matrixB.s0, fma(matrixA.s5, matrixB.s1, fma(matrixA.s9, matrixB.s2, matrixA.sD * matrixB.s3)));
    result.s2 = fma(matrixA.s2, matrixB.s0, fma(matrixA.s6, matrixB.s1, fma(matrixA.sA, matrixB.s2, matrixA.sE * matrixB.s3)));
    result.s3 = fma(matrixA.s3, matrixB.s0, fma(matrixA.s7, matrixB.s1, fma(matrixA.sB, matrixB.s2, matrixA.sF * matrixB.s3)));
    result.s4 = fma(matrixA.s0, matrixB.s4, fma(matrixA.s4, matrixB.s5, fma(matrixA.s8, matrixB.s6, matrixA.sC * matrixB.s7)));
    result.s5 = fma(matrixA.s1, matrixB.s4, fma(matrixA.s5, matrixB.s5, fma(matrixA.s9, matrixB.s6, matrixA.sD * matrixB.s7)));
    result.s6 = fma(matrixA.s2, matrixB.s4, fma(matrixA.s6, matrixB.s5, fma(matrixA.sA, matrixB.s6, matrixA.sE * matrixB.s7)));
    result.s7 = fma(matrixA.s3, matrixB.s4, fma(matrixA.s7, matrixB.s5, fma(matrixA.sB, matrixB.s6, matrixA.sF * matrixB.s7)));
    result.s8 = fma(matrixA.s0, matrixB.s8, fma(matrixA.s4, matrixB.s9, fma(matrixA.s8, matrixB.sA, matrixA.sC * matrixB.sB)));
    result.s9 = fma(matrixA.s1, matrixB.s8, fma(matrixA.s5, matrixB.s9, fma(matrixA.s9, matrixB.sA, matrixA.sD * matrixB.sB)));
    result.sA = fma(matrixA.s2, matrixB.s8, fma(matrixA.s6, matrixB.s9, fma(matrixA.sA, matrixB.sA, matrixA.sE * matrixB.sB)));
    result.sB = fma(matrixA.s3, matrixB.s8, fma(matrixA.s7, matrixB.s9, fma(matrixA.sB, matrixB.sA, matrixA.sF * matrixB.sB)));
    result.sC = fma(matrixA.s0, matrixB.sC, fma(matrixA.s4, matrixB.sD, fma(matrixA.s8, matrixB.sE, matrixA.sC * matrixB.sF)));
    result.sD = fma(matrixA.s1, matrixB.sC, fma(matrixA.s5, matrixB.sD, fma(matrixA.s9, matrixB.sE, matrixA.sD * matrixB.sF)));
    result.sE = fma(matrixA.s2, matrixB.sC, fma(matrixA.s6, matrixB.sD, fma(matrixA.sA, matrixB.sE, matrixA.sE * matrixB.sF)));
    result.sF = fma(matrixA.s3, matrixB.sC, fma(matrixA.s7, matrixB.sD, fma(matrixA.sB, matrixB.sE, matrixA.sF * matrixB.sF)));

    return result;
}