inline float calculate_anti_gravity(float2 gravity, float2 heading)
{
    float dot_p = dot(gravity, heading);
    float mag_p = fast_length(gravity) * fast_length(heading);
    return native_divide(dot_p, mag_p);
}

/**
Performs collision detection using separating axis theorem, and then calculates reactions
for objects when they are found to be colliding. Reactions detemine one "edge" polygon 
and one "vertex" polygon. The vertex polygon has a single vertex adjusted as a reaction. 
The edge object has two vertices adjusted and the adjustments are in oppostie directions, 
which will naturally apply some degree of rotation to the object. For circles/polygon 
collisions, as there is only a single point, circles are always the vertex object.
Circle/circle collisions use a simple distance/radius check.
 */
__kernel void sat_collide(__global int2 *candidates,
                          __global float4 *hulls,
                          __global float2 *hull_frictions,
                          __global int4 *element_tables,
                          __global int4 *hull_flags,
                          __global int4 *vertex_tables,
                          __global float4 *points,
                          __global int2 *edges,
                          __global int *edge_flags,
                          __global float8 *reactions,
                          __global int *reaction_index,
                          __global int *point_reactions,
                          __global float *masses,
                          __global int *counter,
                          float dt)
{
    int gid = get_global_id(0);
    
    int2 current_pair = candidates[gid];
    int b1_id = current_pair.x;
    int b2_id = current_pair.y;
    int4 hull_1_flags = hull_flags[b1_id];
    int4 hull_2_flags = hull_flags[b2_id];
    bool b1s = (hull_1_flags.x & IS_STATIC) !=0;
    bool b2s = (hull_2_flags.x & IS_STATIC) !=0;
    
    bool b1_is_circle = (hull_1_flags.x & IS_CIRCLE) !=0;
    bool b2_is_circle = (hull_2_flags.x & IS_CIRCLE) !=0;

    bool b1_is_polygon = (hull_1_flags.x & IS_POLYGON) !=0;
    bool b2_is_polygon = (hull_2_flags.x & IS_POLYGON) !=0;

    int c_id = b1_is_circle ? b1_id : b2_id;
    int p_id = b1_is_circle ? b2_id : b1_id;

    // todo: it will probably be more performant to have separate kernels for each collision type. There should
    //  be a preliminary kernel that sorts the candidate pairs so they can be run on the right kernel
    if (b1_is_polygon && b2_is_polygon) 
    {
        polygon_collision(b1_id, b2_id, 
            hulls, 
            hull_frictions,
            hull_flags, 
            element_tables, 
            vertex_tables,
            points, 
            edges, 
            edge_flags,
            reactions,
            reaction_index,
            point_reactions,
            masses,
            counter,
            dt);
    }
    else if (b1_is_circle && b2_is_circle) 
    {
        circle_collision(b1_id, b2_id, 
            hulls, 
            hull_frictions,
            hull_flags,
            element_tables, 
            points, 
            reactions,
            reaction_index,
            point_reactions,
            masses,
            counter,
            dt); 
    }
    else 
    {
        polygon_circle_collision(p_id, c_id, 
            hulls, 
            hull_frictions,
            hull_flags, 
            element_tables, 
            vertex_tables,
            points, 
            edges, 
            edge_flags,
            reactions,
            reaction_index,
            point_reactions,
            masses,
            counter,
            dt); 
    }
}

/**
Sorts reaction values in ascending order by point index. Technically the sorting logic is handled 
by way of the reaction scan kernel, which generates the appropriate counts and offsets for each
point. This logic is then a fairly straightforward re-ordering of the buffer in-place. This kernel 
has an implicit assumption that the values in point_reactions have been zeroed out before being
called. These values will have been consumed in a prior call to scan the points for applicable
reactions.
 */
__kernel void sort_reactions(__global float8 *reactions_in,
                             __global float8 *reactions_out,
                             __global int *reaction_index,
                             __global int *point_reactions,
                             __global int *point_offsets)
{
    int gid = get_global_id(0);
    float8 reaction = reactions_in[gid];
    int index = reaction_index[gid];
    int reaction_offset = point_offsets[index];
    int local_offset = atomic_inc(&point_reactions[index]);
    int next = reaction_offset + local_offset;
    reactions_out[next] = reaction;
}

/**
Applies reactions to points by summing all the reactions serially, and then applying the composite 
reaction to the point. 
 */
__kernel void apply_reactions(__global float8 *reactions,
                              __global float4 *points,
                              __global float *anti_gravity,
                              __global int *point_reactions,
                              __global int *point_offsets)
{
    // todo: actual gravity vector should be provided, when it can change this should also be changable
    //  right now it is a static direction. note that magnitude of gravity is not important, only direction
    float2 g = (float2)(0.0f, -1.0f);

    int current_point = get_global_id(0);
    int reaction_count = point_reactions[current_point];

    // exit on non-reactive points
    if (reaction_count == 0) 
    {
        return;
    }
    
    // get the offset into the reaction buffer corresponding to this point
    int reaction_offset = point_offsets[current_point];

    // calculate the cumulative reaction on this point
    float8 reaction = (float8)(0.0f);
    for (int i = 0; i < reaction_count; i++)
    {
        int idx = i + reaction_offset;
        reaction += reactions[idx];
    }
    
    // get the point to be adjusted
    float4 point = points[current_point];

    // store the initial distance and previous position. These are used after
    // adjustment is made to re-adjust the previous position of the point. This
    // is done as a best effort to conserve momentum. 
    float2 initial_tail = point.zw;
    float initial_dist = fast_distance(point.xy, point.zw);

    // apply the cumulative reaction
    point.xy += reaction.s01;

    // apply friction and adjust if necessary 
    float2 friction_test = point.xy + reaction.s45;
    float2 test_velocity = friction_test - point.zw;
    float2 base_velocity = point.xy - point.zw;
    float dot_a = dot(test_velocity, reaction.s45);
    float dot_b = dot(base_velocity, reaction.s45);
    bool sign_a = (dot_a >= 0.0f);
    bool sign_b = (dot_b >= 0.0f);

    // if direction is not reversed by applying friction, it is applied directly. Otherwise it
    // is scaled to ensure it applies only enough force to stop motion completely.
    if (sign_a == sign_b)
    {
        point.xy += reaction.s45;
    }    
    else
    {
        float2 norm = fast_normalize(reaction.s45);
        float mag = fast_length(reaction.s45);
        float2 adjusted_reaction;
        float scale = 1 - native_divide(dot_a, (dot_a + fabs(dot_b)));
        adjusted_reaction = norm * mag * scale;
        point.xy += adjusted_reaction;
    }

    // using the initial data, compared to the new position, calculate the updated previous
    // position to ensure it is equivalent to the initial position delta. This preserves 
    // velocity.
    float2 adjusted_offset = point.xy - initial_tail;
    float new_len = fast_length(adjusted_offset);

    adjusted_offset = new_len == 0.0f 
        ? adjusted_offset 
        : native_divide(adjusted_offset, new_len);

    point.zw = point.xy - initial_dist * adjusted_offset;

     // apply restitution and adjust if necessary 
    float2 restitution_test = point.zw + reaction.s67;
    test_velocity = restitution_test - point.xy;
    base_velocity = point.zw - point.xy;
    dot_a = dot(test_velocity, reaction.s67);
    dot_b = dot(base_velocity, reaction.s67);
    sign_a = (dot_a >= 0.0f);
    sign_b = (dot_b >= 0.0f);

    if (sign_a == sign_b)
    {
        point.zw += reaction.s67;
    }    
    else
    {
        float2 norm = fast_normalize(reaction.s67);
        float mag = fast_length(reaction.s67);
        float2 adjusted_reaction;
        float scale = 1 - native_divide(dot_a, (dot_a + fabs(dot_b)));
        adjusted_reaction = norm * mag * scale;
        point.zw += adjusted_reaction;
    }

    // in addition to velocity preservation, to aid in stabiliy, a non-real force of anti-gravity
    // is modeled to assist in keeping objects from colliding in the direction of gravity. This
    // adjustment is subtle and does not overcome all rigid-body simulation errors, but helps
    // maintain stability with small numbers of stacked objects. 
    float2 heading = reaction.s23;
    float ag = calculate_anti_gravity(g, heading);

    // if anti-gravity would be negative, it means the heading is more in the direction of gravity 
    // than it is against it, so we clamp to 0.
    ag = ag <= 0.0f ? 0.0f : 1.0f;
    //ag = ag >= 0.75f ? 1.0f : 0.0f;


    anti_gravity[current_point] = ag;
    points[current_point] = point;

    // It is important to reset the counts and offsets to 0 after reactions are handled.
    // These reactions are only valid once, for the current frame.
    point_reactions[current_point] = 0;
    point_offsets[current_point] = 0;
}

inline bool any_ag(__global float *anti_gravity,
                   int4 hull_table)
{
    bool result = false;

    int start = hull_table.x;
    int end   = hull_table.y;
	int vert_count = end - start + 1;

    for (int i = 0; i < vert_count; i++)
    {
        int n = start + i;
        float v = anti_gravity[n];
        result = v > 0.0f ? true : result;
    }

    return result;
}
__kernel void move_armatures(__global float4 *hulls,
                             __global float4 *armatures,
                             __global int4 *hull_tables,
                             __global int4 *element_tables,
                             __global int4 *hull_flags,
                             __global float *anti_gravity,
                             __global float4 *points)
{
    int gid = get_global_id(0);
    float4 armature = armatures[gid];
    int4 hull_table = hull_tables[gid];
    int start = hull_table.x;
    int end = hull_table.y;
    int hull_count = end - start + 1;

    float2 diff = (float2)(0.0f);
    bool ag = false;
    float2 last_center = (float2)(0.0f);
    float had_bones = false;
    for (int i = 0; i < hull_count; i++)
    {
        int n = start + i;
        float4 hull = hulls[n];
        int4 hull_flag = hull_flags[n];
        int4 element_table = element_tables[n];
        bool no_bones = (hull_flag.x & NO_BONES) !=0;

        if (!no_bones) had_bones = true;

        float2 center_a = calculate_centroid(points, element_table);
        last_center = center_a;
        float2 diffa = center_a - hull.xy;
        diff += diffa;
        bool has_ag = any_ag(anti_gravity, element_table);
        ag = has_ag ? true : ag;
    }

    if (had_bones) armature.xy += diff;
    else armature.xy = last_center;
    armature.w = ag ? armature.y : armature.w;
    armatures[gid] = armature;
}