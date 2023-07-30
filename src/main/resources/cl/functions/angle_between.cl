inline float angle_between(float4 line1, float4 line2)
{
    float l1x1 = line1.x;
    float l1y1 = line1.y;
    float l1x2 = line1.z;
    float l1y2 = line1.w;

    float l2x1 = line2.x;
    float l2y1 = line2.y;
    float l2x2 = line2.z;
    float l2y2 = line2.w;
    
    float angle1 = atan2(l1y1 - l1y2, l1x1 - l1x2);
    float angle2 = atan2(l2y1 - l2y2, l2x1 - l2x2);
    return angle1 - angle2;
}