inline float16 matrix_mul(float16 matrixA, float16 matrixB) 
{
    float16 result;

    result.s0 = mad(matrixA.s0, matrixB.s0, mad(matrixA.s4, matrixB.s1, mad(matrixA.s8, matrixB.s2, matrixA.sC * matrixB.s3)));
    result.s1 = mad(matrixA.s1, matrixB.s0, mad(matrixA.s5, matrixB.s1, mad(matrixA.s9, matrixB.s2, matrixA.sD * matrixB.s3)));
    result.s2 = mad(matrixA.s2, matrixB.s0, mad(matrixA.s6, matrixB.s1, mad(matrixA.sA, matrixB.s2, matrixA.sE * matrixB.s3)));
    result.s3 = mad(matrixA.s3, matrixB.s0, mad(matrixA.s7, matrixB.s1, mad(matrixA.sB, matrixB.s2, matrixA.sF * matrixB.s3)));
    result.s4 = mad(matrixA.s0, matrixB.s4, mad(matrixA.s4, matrixB.s5, mad(matrixA.s8, matrixB.s6, matrixA.sC * matrixB.s7)));
    result.s5 = mad(matrixA.s1, matrixB.s4, mad(matrixA.s5, matrixB.s5, mad(matrixA.s9, matrixB.s6, matrixA.sD * matrixB.s7)));
    result.s6 = mad(matrixA.s2, matrixB.s4, mad(matrixA.s6, matrixB.s5, mad(matrixA.sA, matrixB.s6, matrixA.sE * matrixB.s7)));
    result.s7 = mad(matrixA.s3, matrixB.s4, mad(matrixA.s7, matrixB.s5, mad(matrixA.sB, matrixB.s6, matrixA.sF * matrixB.s7)));
    result.s8 = mad(matrixA.s0, matrixB.s8, mad(matrixA.s4, matrixB.s9, mad(matrixA.s8, matrixB.sA, matrixA.sC * matrixB.sB)));
    result.s9 = mad(matrixA.s1, matrixB.s8, mad(matrixA.s5, matrixB.s9, mad(matrixA.s9, matrixB.sA, matrixA.sD * matrixB.sB)));
    result.sA = mad(matrixA.s2, matrixB.s8, mad(matrixA.s6, matrixB.s9, mad(matrixA.sA, matrixB.sA, matrixA.sE * matrixB.sB)));
    result.sB = mad(matrixA.s3, matrixB.s8, mad(matrixA.s7, matrixB.s9, mad(matrixA.sB, matrixB.sA, matrixA.sF * matrixB.sB)));
    result.sC = mad(matrixA.s0, matrixB.sC, mad(matrixA.s4, matrixB.sD, mad(matrixA.s8, matrixB.sE, matrixA.sC * matrixB.sF)));
    result.sD = mad(matrixA.s1, matrixB.sC, mad(matrixA.s5, matrixB.sD, mad(matrixA.s9, matrixB.sE, matrixA.sD * matrixB.sF)));
    result.sE = mad(matrixA.s2, matrixB.sC, mad(matrixA.s6, matrixB.sD, mad(matrixA.sA, matrixB.sE, matrixA.sE * matrixB.sF)));
    result.sF = mad(matrixA.s3, matrixB.sC, mad(matrixA.s7, matrixB.sD, mad(matrixA.sB, matrixB.sE, matrixA.sF * matrixB.sF)));

    return result;
}
