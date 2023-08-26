inline void circle_collision(int b1_id, int b2_id,
                             __global float4 *hulls,
                             __global int4 *element_tables,
                             __global float4 *points,
                             __global float2 *reactions,
                             __global int *reaction_index,
                             __global int *point_reactions,
                             __global int *counter)
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

    float2 reaction = depth * normal;
    float2 offset1 = -0.5f * reaction;
    float2 offset2 = offset1 * -1.0f;

    int i = atomic_inc(&counter[0]);
    int j = atomic_inc(&counter[0]);

    reactions[i] = offset1;
    reactions[j] = offset2;

    reaction_index[i] = hull_1_table.x;
    reaction_index[j] = hull_2_table.x;

    atomic_inc(&point_reactions[hull_1_table.x]);
    atomic_inc(&point_reactions[hull_2_table.x]);
}