inline void circle_collision(int b1_id, int b2_id,
                             __global float4 *hulls,
                             __global int4 *hull_flags,
                             __global int4 *element_tables,
                             __global float4 *points,
                             __global float4 *reactions,
                             __global int *reaction_index,
                             __global int *point_reactions,
                             __global float *masses,
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

    float2 h1_dir = hull_2.xy - hull_1.xy;
    float2 h2_dir = hull_1.xy - hull_2.xy;

    normal = fast_normalize(h1_dir);
    depth = radii - _distance;
    
    int4 vo_f = hull_flags[b1_id];
    int4 eo_f = hull_flags[b2_id];

    float mass1 = masses[vo_f.y];
    float mass2 = masses[eo_f.y];

    float total_mass = mass1 + mass2;

    float mag1 = mass2 / total_mass;
    float mag2 = mass1 / total_mass;

    float2 reaction = depth * normal;
    float2 offset1 = -mag1 * reaction;
    float2 offset2 = mag2 * reaction;

    float4 offset1_4d;
    offset1_4d.xy = offset1;
    offset1_4d.zw = h1_dir;

    float4 offset2_4d;
    offset2_4d.xy = offset2;
    offset2_4d.zw = h2_dir;

    int i = atomic_inc(&counter[0]);
    int j = atomic_inc(&counter[0]);

    reactions[i] = offset1_4d;
    reactions[j] = offset2_4d;

    reaction_index[i] = hull_1_table.x;
    reaction_index[j] = hull_2_table.x;

    atomic_inc(&point_reactions[hull_1_table.x]);
    atomic_inc(&point_reactions[hull_2_table.x]);
}