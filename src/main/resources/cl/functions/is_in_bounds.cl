/**
Determines if a given bounding box is within the current spatial index boundary.
 */
inline bool is_in_bounds(float4 a, float x, float y, float w, float h)
{
    return a.s0 < x + w
        && a.s0 + a.s2 > x
        && a.s1 < y + h
        && a.s1 + a.s3 > y;
}


/**
Determines if a given point is within a provided boundary.
 */
inline bool is_point_in_bounds(float2 p, float x, float y, float w, float h)
{
    return p.x >= x 
        && p.x <= x + w 
        && p.y >= y 
        && p.y <= y + h;
}