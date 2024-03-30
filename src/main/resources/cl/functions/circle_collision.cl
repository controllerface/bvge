inline void circle_collision(int hull_1_id, 
                             int hull_2_id,
                             __global float4 *hulls,
                             __global float2 *hull_frictions,
                             __global int4 *hull_flags,
                             __global int4 *element_tables,
                             __global float4 *points,
                             __global float4 *reactions_A,
                             __global float4 *reactions_B,
                             __global int *reaction_index,
                             __global int *reaction_counts,
                             __global float *masses,
                             __global int *counter,
                             float dt)
{
    float4 hull_1 = hulls[hull_1_id];
    float4 hull_2 = hulls[hull_2_id];
    float center_distance = fast_distance(hull_1.xy, hull_2.xy);
    float radii_sum = hull_1.w + hull_2.w;

    if(center_distance >= radii_sum) return;
    
    float collision_depth = radii_sum - center_distance;

    int4 hull_1_table = element_tables[hull_1_id];
    int4 hull_2_table = element_tables[hull_2_id];
    float4 hull_1_center = points[hull_1_table.x];
    float4 hull_2_center = points[hull_2_table.x];
    int4 hull_1_flags = hull_flags[hull_1_id];
    int4 hull_2_flags = hull_flags[hull_2_id];
    float2 hull_1_phys = hull_frictions[hull_1_id];
    float2 hull_2_phys = hull_frictions[hull_2_id];
    float hull_1_mass = masses[hull_1_flags.y];
    float hull_2_mass = masses[hull_2_flags.y];

    float2 hull_1_ooposing = hull_2.xy - hull_1.xy;
    float2 hull_2_opposing = hull_1.xy - hull_2.xy;
    float2 collision_normal = fast_normalize(hull_2_opposing);
    float total_mass = hull_1_mass + hull_2_mass;
    float hull_1_magnitude = native_divide(hull_2_mass, total_mass);
    float hull_2_magnitude = native_divide(hull_1_mass, total_mass);
    float2 collision_reaction = collision_depth * collision_normal;
    float2 hull_1_collision = hull_1_magnitude * collision_reaction;
    float2 hull_2_collision = -hull_2_magnitude * collision_reaction;

    float2 collision_vector = collision_normal * collision_depth;
    float2 hull_1_diff = hull_1_center.xy - hull_1_center.zw;
    float2 hull_2_diff = hull_2_center.xy - hull_2_center.zw;;
    float2 hull_1_velocity = native_divide(hull_1_diff, dt);
    float2 hull_2_velocity = native_divide(hull_2_diff, dt);
    float2 hull_1_rel_vel = hull_1_velocity - collision_vector;
    float2 hull_2_rel_vel = hull_2_velocity - collision_vector;
    float friction_coefficient = max(hull_1_phys.x, hull_2_phys.x);
    float2 hull_1_tangent = hull_1_rel_vel - dot(hull_1_rel_vel, collision_normal) * collision_normal;
    float2 hull_2_tangent = hull_2_rel_vel - dot(hull_2_rel_vel, collision_normal) * collision_normal;
    hull_1_tangent = fast_normalize(hull_1_tangent);
    hull_2_tangent = fast_normalize(hull_2_tangent);
    float2 hull_1_friction = (-friction_coefficient * hull_1_tangent) * hull_1_magnitude;
    float2 hull_2_friction = (-friction_coefficient * hull_2_tangent) * -hull_2_magnitude;

    float2 hull_1_applied = hull_1_center.xy + hull_1_collision;
    float2 hull_2_applied = hull_2_center.xy + hull_2_collision;
    float2 hull_1_applied_diff = hull_1_applied - hull_1_center.zw;
    float2 hull_2_applied_diff = hull_2_applied - hull_2_center.zw;
    float2 hull_1_applied_vel = native_divide(hull_1_applied_diff, dt);
    float2 hull_2_applied_vel = native_divide(hull_2_applied_diff, dt);
    float restituion_coefficient = max(hull_1_phys.y, hull_2_phys.y);
    float2 collision_invert = collision_normal * -1;
    float2 hull_1_restitution = restituion_coefficient * dot(hull_1_applied_vel, collision_normal) * collision_normal;
    float2 hull_2_restitution = restituion_coefficient * dot(hull_2_applied_vel, collision_invert) * collision_invert;
    
    float4 hull_1_reactions_A = (float4)(hull_1_collision.xy, hull_1_ooposing.xy);
    float4 hull_1_reactions_B = (float4)(hull_1_friction.xy, hull_1_restitution.xy);
    float4 hull_2_reactions_A = (float4)(hull_2_collision.xy, hull_2_opposing.xy);
    float4 hull_2_reactions_B = (float4)(hull_2_friction.xy, hull_2_restitution.xy);
    int hull_1_index = atomic_inc(&counter[0]);
    int hull_2_index = atomic_inc(&counter[0]);
    reactions_A[hull_1_index] = hull_1_reactions_A;
    reactions_B[hull_1_index] = hull_1_reactions_B;
    reactions_A[hull_2_index] = hull_2_reactions_A;
    reactions_B[hull_2_index] = hull_2_reactions_B;
    reaction_index[hull_1_index] = hull_1_table.x;
    reaction_index[hull_2_index] = hull_2_table.x;
    atomic_inc(&reaction_counts[hull_1_table.x]);
    atomic_inc(&reaction_counts[hull_2_table.x]);
}