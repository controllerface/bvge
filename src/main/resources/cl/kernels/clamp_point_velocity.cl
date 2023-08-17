__kernel void clamp_point_velocity(__global float4 *points)
{
    int gid = get_global_id(0);
    float4 point = points[gid];

    float constraint = 8.0f;;
    
    // extract just the current vertex info for processing
    float2 p1_v = point.zw;
    float2 p2_v = point.xy;
    
    // calculate the normalized direction of separation
    float2 sub = p2_v - p1_v;
    float len = length(sub);

    if (len > constraint) 
    {
        printf("debug: len: %f", len);
        float ratio =  constraint / len;
        float newX = (float) (p2_v.x - sub.x * ratio);
        float newY = (float) (p2_v.y - sub.y * ratio);
        point.z = newX;
        point.w = newY;
        points[gid] = point;
    }
}