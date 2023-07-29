
// todo: move to float 4, extents

inline bool do_bounds_intersect(float4 a, float4 b)
{
    return a.s0 < b.s0 + b.s2
        && a.s0 + a.s2 > b.s0
        && a.s1 < b.s1 + b.s3
        && a.s1 + a.s3 > b.s1;
}
