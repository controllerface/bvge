/**
Determines if a given bounding box is within the current spatial index boundary.
 */
inline bool isInBounds(float16 a, float x, float y, float w, float h)
{
    return a.s0 < x + w
        && a.s0 + a.s2 > x
        && a.s1 < y + h
        && a.s1 + a.s3 > y;
}
