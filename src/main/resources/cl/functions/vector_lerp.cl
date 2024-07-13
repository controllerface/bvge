inline float4 float4_lerp(float4 a, float4 b, float t) 
{
    return mad(b - a, t, a);
}

inline float2 float2_lerp(float2 a, float2 b, float t) 
{
    return mad(b - a, t, a);
}
