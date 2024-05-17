/**
Handles collision between two circular hulls
 */
inline void circle_collision(int hull_1_id, 
                             int hull_2_id,
                             __global float4 *hulls,
                             __global float2 *hull_scales,
                             __global float *hull_frictions,
                             __global float *hull_restitutions,
                             __global int *hull_armature_ids,
                             __global int *hull_flags,
                             __global int2 *hull_point_tables,
                             __global float4 *points,
                             __global int *point_flags,
                             __global float8 *reactions,
                             __global int *reaction_index,
                             __global int *reaction_counts,
                             __global float *masses,
                             __global int *counter,
                             float dt)
{
    float4 hull_1 = hulls[hull_1_id];
    float4 hull_2 = hulls[hull_2_id];
    float hull_1_radius = hull_scales[hull_1_id].y;
    float hull_2_radius = hull_scales[hull_2_id].y;

    // collision detection
    float center_distance = fast_distance(hull_1.xy, hull_2.xy);
    float radii_sum = hull_1_radius + hull_2_radius;
    if (center_distance >= radii_sum) return;
    
    int2 hull_1_table = hull_point_tables[hull_1_id];
    int2 hull_2_table = hull_point_tables[hull_2_id];
    int hull_1_flags = hull_flags[hull_1_id];
    int hull_2_flags = hull_flags[hull_2_id];
    float4 hull_1_center = points[hull_1_table.x];
    float4 hull_2_center = points[hull_2_table.x];
    int hull_1_armature_id = hull_armature_ids[hull_1_id];
    int hull_2_armature_id = hull_armature_ids[hull_2_id];
    float hull_1_friction = hull_frictions[hull_1_id];
    float hull_2_friction = hull_frictions[hull_2_id];
    float hull_1_restitution = hull_restitutions[hull_1_id];
    float hull_2_restitution = hull_restitutions[hull_2_id];
    float hull_1_mass = masses[hull_1_armature_id];
    float hull_2_mass = masses[hull_2_armature_id];

    // cursor collision causes early exit
    bool cursor_1 = (hull_1_flags & IS_CURSOR) !=0;
    bool cursor_2 = (hull_2_flags & IS_CURSOR) !=0;
    bool any_cursor = (cursor_1 || cursor_2);
    if (any_cursor)
    {
        if (cursor_1)
        {
            hull_2_flags |= CURSOR_OVER;            
            hull_flags[hull_2_id] = hull_2_flags;
        }
        else
        {
            hull_1_flags |= CURSOR_OVER;            
            hull_flags[hull_1_id] = hull_1_flags;
        }
        return;
    }

    // todo: this should be conditional on liquid type
    hull_1_flags |= TOUCH_ALIKE;
    hull_2_flags |= TOUCH_ALIKE;

    // collision reaction and opposing direction calculation
    float2 hull_1_opposing = hull_2.xy - hull_1.xy;
    float2 hull_2_opposing = hull_1.xy - hull_2.xy;
    float2 collision_normal = fast_normalize(hull_2_opposing);
    float collision_depth = radii_sum - center_distance;
    float total_mass = hull_1_mass + hull_2_mass;
    float hull_1_magnitude = native_divide(hull_2_mass, total_mass);
    float hull_2_magnitude = native_divide(hull_1_mass, total_mass);
    float2 collision_reaction = collision_depth * collision_normal;
    float2 hull_1_collision = hull_1_magnitude * collision_reaction;
    float2 hull_2_collision = -hull_2_magnitude * collision_reaction;

    // friction
    float2 collision_vector = collision_normal * collision_depth;
    float2 hull_1_diff = hull_1_center.xy - hull_1_center.zw;
    float2 hull_2_diff = hull_2_center.xy - hull_2_center.zw;;
    float2 hull_1_velocity = native_divide(hull_1_diff, dt);
    float2 hull_2_velocity = native_divide(hull_2_diff, dt);
    float2 hull_1_rel_vel = hull_1_velocity - collision_vector;
    float2 hull_2_rel_vel = hull_2_velocity - collision_vector;
    float friction_coefficient = max(hull_1_friction, hull_2_friction);
    float2 hull_1_tangent = hull_1_rel_vel - dot(hull_1_rel_vel, collision_normal) * collision_normal;
    float2 hull_2_tangent = hull_2_rel_vel - dot(hull_2_rel_vel, collision_normal) * collision_normal;
    hull_1_tangent = fast_normalize(hull_1_tangent);
    hull_2_tangent = fast_normalize(hull_2_tangent);
    float2 hull_1_friction_f = (-friction_coefficient * hull_1_tangent) * hull_1_magnitude;
    float2 hull_2_friction_f = (-friction_coefficient * hull_2_tangent) * -hull_2_magnitude;

    // restitution
    float2 hull_1_applied = hull_1_center.xy + hull_1_collision;
    float2 hull_2_applied = hull_2_center.xy + hull_2_collision;
    float2 hull_1_applied_diff = hull_1_applied - hull_1_center.zw;
    float2 hull_2_applied_diff = hull_2_applied - hull_2_center.zw;
    float2 hull_1_applied_vel = native_divide(hull_1_applied_diff, dt);
    float2 hull_2_applied_vel = native_divide(hull_2_applied_diff, dt);
    float restituion_coefficient = max(hull_1_restitution, hull_2_restitution);
    float2 collision_invert = collision_normal * -1;
    float2 hull_1_restitution_f = restituion_coefficient * dot(hull_1_applied_vel, collision_normal) * collision_normal;
    float2 hull_2_restitution_f = restituion_coefficient * dot(hull_2_applied_vel, collision_invert) * collision_invert;

    // store results
    float8 hull_1_reactions = (float8)(hull_1_collision, hull_1_opposing, hull_1_friction_f, hull_1_restitution_f);
    float8 hull_2_reactions = (float8)(hull_2_collision, hull_2_opposing, hull_2_friction_f, hull_2_restitution_f);
    int hull_1_reaction_index = atomic_inc(&counter[0]);
    int hull_2_reaction_index = atomic_inc(&counter[0]);
    reactions[hull_1_reaction_index] = hull_1_reactions;
    reactions[hull_2_reaction_index] = hull_2_reactions;
    reaction_index[hull_1_reaction_index] = hull_1_table.x;
    reaction_index[hull_2_reaction_index] = hull_2_table.x;
    atomic_inc(&reaction_counts[hull_1_table.x]);
    atomic_inc(&reaction_counts[hull_2_table.x]);
    hull_flags[hull_1_id] = hull_1_flags;
    hull_flags[hull_2_id] = hull_2_flags;
}
