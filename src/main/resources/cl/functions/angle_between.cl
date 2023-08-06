/**
Calculates the angle between two lines. Each line is packed into a float4, the xy
components containing the start point of the line, and the zw components containing 
the end point. The angle returned is in radians. 
 */
inline float angle_between(float4 line1, float4 line2)
{
    return atan2(line1.y - line1.w, line1.x - line1.z) - atan2(line2.y - line2.w, line2.x - line2.z);
}