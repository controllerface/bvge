inline float16 matrix_mul_affine(float16 matrixA, float16 matrixB) 
{
    float16 result;

    float a00 = matrixA.s0;
    float a01 = matrixA.s1;
    float a02 = matrixA.s2;
    float a10 = matrixA.s4;
    float a11 = matrixA.s5;
    float a12 = matrixA.s6;
    float a20 = matrixA.s8;
    float a21 = matrixA.s9;
    float a22 = matrixA.sA;

    float b00 = matrixB.s0;
    float b01 = matrixB.s1;
    float b02 = matrixB.s2;
    float b10 = matrixB.s4;
    float b11 = matrixB.s5;
    float b12 = matrixB.s6;
    float b20 = matrixB.s8;
    float b21 = matrixB.s9;
    float b22 = matrixB.sA;
    float b30 = matrixB.sC;
    float b31 = matrixB.sD;
    float b32 = matrixB.sE;

    result.s0 = mad(a00, b00, mad(a10, b01, a20 * b02));
    result.s1 = mad(a01, b00, mad(a11, b01, a21 * b02)); 
    result.s2 = mad(a02, b00, mad(a12, b01, a22 * b02)); 
    result.s3 = matrixA.s3; 
    result.s4 = mad(a00, b10, mad(a10, b11, a20 * b12)); 
    result.s5 = mad(a01, b10, mad(a11, b11, a21 * b12)); 
    result.s6 = mad(a02, b10, mad(a12, b11, a22 * b12)); 
    result.s7 = matrixA.s7; 
    result.s8 = mad(a00, b20, mad(a10, b21, a20 * b22)); 
    result.s9 = mad(a01, b20, mad(a11, b21, a21 * b22)); 
    result.sA = mad(a02, b20, mad(a12, b21, a22 * b22)); 
    result.sB = matrixA.sB; 
    result.sC = mad(a00, b30, mad(a10, b31, mad(a20, b32, matrixA.sC))); 
    result.sD = mad(a01, b30, mad(a11, b31, mad(a21, b32, matrixA.sD))); 
    result.sE = mad(a02, b30, mad(a12, b31, mad(a22, b32, matrixA.sE))); 
    result.sF = matrixA.sF; 

    return result;
}
