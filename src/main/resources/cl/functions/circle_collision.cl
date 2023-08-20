inline void circle_collision(int b1_id, int b2_id,
                             __global float4 *hulls,
                             __global int4 *element_tables,
                             __global float4 *points)
{
    float4 hull_1 = hulls[b1_id];
    float4 hull_2 = hulls[b2_id];
    int4 hull_1_table = element_tables[b1_id];
    int4 hull_2_table = element_tables[b2_id];

    float2 normal;
    float depth = 0;
    float _distance = distance(hull_1.xy, hull_2.xy);
    float radii = hull_1.w + hull_2.w;
    if(_distance >= radii)
    {
        return;
    }

    float2 sub = hull_2.xy - hull_1.xy;
    normal = normalize(sub);
    depth = radii - _distance;

    float2 reaction = normal * (float2)(depth);
    float2 offset1 = reaction * (float2)(-0.5);
    float2 offset2 = reaction * (float2)(0.5);

    float4 vert1 = points[hull_1_table.x];
    float4 vert2 = points[hull_2_table.x];
    
    vert1.xy += offset1;
    vert2.xy += offset2;

    points[hull_1_table.x] = vert1;
    points[hull_2_table.x] = vert2;
}