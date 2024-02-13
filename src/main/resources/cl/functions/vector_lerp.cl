inline float4 vector_lerp(float4 a, float4 b, float t) 
{
    return fma(b - a, t, a);
}