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

    float2 reaction = normal * (float2)(depth);
    float2 offset1 = reaction * (float2)(-0.5);
    float2 offset2 = reaction * (float2)(0.5);

    // float4 vert1 = points[hull_1_table.x];
    // float4 vert2 = points[hull_2_table.x];

    // float2 e1 = vert1.xy;
    // float2 e2 = vert2.xy;

    // float2 e1_p = vert1.zw;
    // float2 e2_p = vert2.zw;

    // float e1_dist = distance(e1, e1_p);
    // float e2_dist = distance(e2, e2_p);
    
    // vert1.xy += offset1;
    // vert2.xy += offset2;

    // float2 e1_diff_2 = vert1.xy - e1_p;
    // float2 e2_diff_2 = vert2.xy - e2_p;

    // float new_len_e1 = length(e1_diff_2);
    // float new_len_e2 = length(e2_diff_2);

    // if (new_len_e1 != 0.0)
    // {
    //     e1_diff_2 /= new_len_e1;
    //     vert1.zw = vert1.xy - e1_dist * e1_diff_2;
    // }

    // if (new_len_e2 != 0.0)
    // {
    //     e2_diff_2 /= new_len_e2;
    //     vert2.zw = vert2.xy - e2_dist * e2_diff_2;
    // }

    int i = atomic_inc(&counter[0]);
    int j = atomic_inc(&counter[0]);

    reactions[i] = offset1;
    reactions[j] = offset2;

    reaction_index[i] = hull_1_table.x;
    reaction_index[j] = hull_2_table.x;

    atomic_inc(&point_reactions[hull_1_table.x]);
    atomic_inc(&point_reactions[hull_2_table.x]);

    // todo: increment an atomic per-point counter to indicate how many reactions each point has

    // todo: below will be defferred to a later kernel

    // points[hull_1_table.x] = vert1;
    // points[hull_2_table.x] = vert2;
}