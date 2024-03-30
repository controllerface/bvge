inline void circle_collision(int b1_id, int b2_id,
                             __global float4 *hulls,
                             __global float2 *hull_frictions,
                             __global int4 *hull_flags,
                             __global int4 *element_tables,
                             __global float4 *points,
                             __global float4 *reactions,
                             __global float4 *reactions2,
                             __global int *reaction_index,
                             __global int *point_reactions,
                             __global float *masses,
                             __global int *counter,
                             float dt)
{
    float4 hull_1 = hulls[b1_id];
    float4 hull_2 = hulls[b2_id];
    float _distance = fast_distance(hull_1.xy, hull_2.xy);
    float radii = hull_1.w + hull_2.w;

    if(_distance >= radii)
    {
        return;
    }

    int4 hull_1_table = element_tables[b1_id];
    int4 hull_2_table = element_tables[b2_id];
    float2 h1_dir = hull_2.xy - hull_1.xy;
    float2 h2_dir = hull_1.xy - hull_2.xy;
    float4 p1 = points[hull_1_table.x];
    float4 p2 = points[hull_2_table.x];
    float2 normal = fast_normalize(h2_dir);
    float depth = radii - _distance;
    int4 vo_f = hull_flags[b1_id];
    int4 eo_f = hull_flags[b2_id];
    float2 vo_phys = hull_frictions[b1_id];
    float2 eo_phys = hull_frictions[b2_id];

    float mass1 = masses[vo_f.y];
    float mass2 = masses[eo_f.y];
    float total_mass = mass1 + mass2;
    float mag1 = native_divide(mass2, total_mass);
    float mag2 = native_divide(mass1, total_mass);

    float2 collision_vector = normal * depth;
    float2 e1_dir = p1.xy - p1.zw;
    float2 e2_dir = p2.xy - p2.zw;;
    float2 e1_v = native_divide(e1_dir, dt);
    float2 e2_v = native_divide(e2_dir, dt);
    float2 e1_rel = e1_v - collision_vector;
    float2 e2_rel = e2_v - collision_vector;
    float mu = max(vo_phys.x, eo_phys.x);
    float2 e1_tan = e1_rel - dot(e1_rel, normal) * normal;
    float2 e2_tan = e2_rel - dot(e2_rel, normal) * normal;
    e1_tan = fast_normalize(e1_tan);
    e2_tan = fast_normalize(e2_tan);
    float2 e1_fric = (-mu * e1_tan) * mag1;
    float2 e2_fric = (-mu * e2_tan) * -mag2;

    float2 reaction = depth * normal;
    float2 offset1 = mag1 * reaction;
    float2 offset2 = -mag2 * reaction;
    float2 e1_n = p1.xy + offset1;
    float2 e2_n = p2.xy + offset2;
    float2 e1_dir_n = e1_n - p1.zw;
    float2 e2_dir_n = e2_n - p2.zw;
    float2 e1_vn = native_divide(e1_dir_n, dt);
    float2 e2_vn = native_divide(e2_dir_n, dt);
    float ru = max(vo_phys.y, eo_phys.y);
    float2 normal_inv = normal * -1;
    float2 e1_rest = ru * dot(e1_vn, normal) * normal;
    float2 e2_rest = ru * dot(e2_vn, normal_inv) * normal_inv;
    
    float4 offset1_4d = (float4)(offset1.xy, h1_dir.xy);
    float4 offset1_4d2 = (float4)(e1_fric.xy, e1_rest.xy);
    float4 offset2_4d = (float4)(offset2.xy, h2_dir.xy);
    float4 offset2_4d2 = (float4)(e2_fric.xy, e2_rest.xy);

    int i = atomic_inc(&counter[0]);
    int j = atomic_inc(&counter[0]);

    reactions[i] = offset1_4d;
    reactions[j] = offset2_4d;
    reactions2[i] = offset1_4d2;
    reactions2[j] = offset2_4d2;
    reaction_index[i] = hull_1_table.x;
    reaction_index[j] = hull_2_table.x;

    atomic_inc(&point_reactions[hull_1_table.x]);
    atomic_inc(&point_reactions[hull_2_table.x]);
}