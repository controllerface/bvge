inline float4 quaternion_lerp(float4 a, float4 b, float factor) 
{
    float4 dest;
    float cosom = fma(a.x, b.x, fma(a.y, b.y, fma(a.z, b.z, a.w * b.w)));
    float scale0 = 1.0F - factor;
    float scale1 = cosom >= 0.0F ? factor : -factor;
    
    dest.x = fma(scale0, a.x, scale1 * b.x);
    dest.y = fma(scale0, a.y, scale1 * b.y);
    dest.z = fma(scale0, a.z, scale1 * b.z);
    dest.w = fma(scale0, a.w, scale1 * b.w);

    float s = native_rsqrt(fma(dest.x, dest.x, fma(dest.y, dest.y, fma(dest.z, dest.z, dest.w * dest.w))));
    dest *= s;

    return dest;
}