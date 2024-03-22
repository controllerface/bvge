inline float4 quaternion_lerp(float4 a, float4 b, float factor) 
{
    float4 dest;
    float cosom = mad(a.x, b.x, mad(a.y, b.y, mad(a.z, b.z, a.w * b.w)));
    float scale0 = 1.0F - factor;
    float scale1 = cosom >= 0.0F ? factor : -factor;
    
    dest.x = mad(scale0, a.x, scale1 * b.x);
    dest.y = mad(scale0, a.y, scale1 * b.y);
    dest.z = mad(scale0, a.z, scale1 * b.z);
    dest.w = mad(scale0, a.w, scale1 * b.w);

    float s = native_rsqrt(mad(dest.x, dest.x, mad(dest.y, dest.y, mad(dest.z, dest.z, dest.w * dest.w))));
    dest *= s;

    return dest;
}