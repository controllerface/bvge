inline float calculate_anti_gravity(float2 gravity, float2 heading)
{
    float dot_p = dot(gravity, heading);
    float mag_p = length(gravity) * length(heading);
    return dot_p / mag_p;
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
                          __global int4 *element_tables,
                          __global int2 *hull_flags,
                          __global float4 *points,
                          __global float4 *edges,
                          __global float2 *reactions,
                          __global int *reaction_index,
                          __global int *point_reactions,
                          __global int *counter)
{
    int gid = get_global_id(0);
    
    int2 current_pair = candidates[gid];
    int b1_id = current_pair.x;
    int b2_id = current_pair.y;
    int2 hull_1_flags = hull_flags[b1_id];
    int2 hull_2_flags = hull_flags[b2_id];
    bool b1s = (hull_1_flags.x & 0x01) !=0;
    bool b2s = (hull_2_flags.x & 0x01) !=0;
    
    if (b1s && b2s) // no collisions between static objects todo: probably can weed these out earlier, during aabb checks
    {
        return;
    }

    bool b1_is_circle = (hull_1_flags.x & 0x02) !=0;
    bool b2_is_circle = (hull_2_flags.x & 0x02) !=0;

    bool b1_is_polygon = (hull_1_flags.x & 0x04) !=0;
    bool b2_is_polygon = (hull_2_flags.x & 0x04) !=0;

    int c_id = b1_is_circle ? b1_id : b2_id;
    int p_id = b1_is_circle ? b2_id : b1_id;

    // todo: it will probably be more performant to have separate kernels for each collision type. There should
    //  be a preliminary kernel that sorts the candidate pairs so they can be run on the right kernel
    if (b1_is_polygon && b2_is_polygon) 
    {
        polygon_collision(b1_id, b2_id, 
            hulls, 
            hull_flags, 
            element_tables, 
            points, 
            edges, 
            reactions,
            reaction_index,
            point_reactions,
            counter); 
    }
    else if (b1_is_circle && b2_is_circle) 
    {
        circle_collision(b1_id, b2_id, 
            hulls, 
            element_tables, 
            points, 
            reactions,
            reaction_index,
            point_reactions,
            counter); 
    }
    else 
    {
        polygon_circle_collision(p_id, c_id, 
            hulls, 
            hull_flags, 
            element_tables, 
            points, 
            edges, 
            reactions,
            reaction_index,
            point_reactions,
            counter); 
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
__kernel void sort_reactions(__global float2 *reactions,
                             __global int *reaction_index,
                             __global int *point_reactions,
                             __global int *point_offsets)
{
    int gid = get_global_id(0);
    
    float2 reaction = reactions[gid];
    int index = reaction_index[gid];

    int reaction_offset = point_offsets[index];
    int local_offset = atomic_inc(&point_reactions[index]);

    // this barrier is extremely important, it ensures all threads have read their reactions before 
    // moving them to their correct positions.
    barrier(CLK_GLOBAL_MEM_FENCE);

    int next = reaction_offset + local_offset;

    reactions[next] = reaction;
    reaction_index[next] = index;
}

/**
Applies reactions to points by summing all the reactions serially, and then apoplying the composite 
reaction to the point. 
 */
__kernel void apply_reactions(__global float2 *reactions,
                              __global float4 *points,
                              __global float *anti_gravity,
                              __global int *point_reactions,
                              __global int *point_offsets)
{
    int gid = get_global_id(0);
    int reaction_count = point_reactions[gid];

    if (reaction_count == 0) return;

    int offset = point_offsets[gid];
    float4 point = points[gid];

    float2 e1_p = point.zw;
    float e1_dist = distance(point.xy, e1_p);

    float2 reaction = (float2)(0.0, 0.0);

    float x_sum = 0;
    float y_sum = 0;

    for (int i = 0; i < reaction_count; i++)
    {
        int idx = i + offset;
        float2 reaction_i = reactions[idx];
        x_sum += reaction_i.x;
        y_sum += reaction_i.y;
        reaction += reaction_i;
    }

    float x = x_sum / reaction_count;
    float y = y_sum / reaction_count;
    
    // point.x += x;
    // point.y += y;

    point.xy += reaction;

    float2 e1_diff_2 = point.xy - e1_p;
    float new_len_e1 = length(e1_diff_2);

    if (new_len_e1 != 0.0)
    {
        e1_diff_2 /= new_len_e1;
        point.zw = point.xy - e1_dist * e1_diff_2;
    }

    // todo: calculate direction of movment relative to gravity and accumulate anti-grav for this point

    float2 g = (float2)(0.0, -1.0);
    float2 h = point.xy - point.zw;
    //float2 h = (float2)(x, y);
    float ag = calculate_anti_gravity(g, h);

    if (ag < 0.0f) ag = 0.0f;

    anti_gravity[gid] = ag;
    points[gid] = point;

    // It is important to reset the counts and offsets to 0 after reactions are handled.
    // These reactions are only valid once, for the current frame.
    point_reactions[gid] = 0;
    point_offsets[gid] = 0;
}

__kernel void move_armatures(__global float4 *hulls,
                             __global float4 *armatures,
                             __global int2 *hull_tables,
                             __global int4 *element_tables,
                             __global int2 *hull_flags,
                             __global float4 *points)
{
    int gid = get_global_id(0);
    float4 armature = armatures[gid];
    int2 hull_table = hull_tables[gid];
    int start = hull_table.x;
    int end = hull_table.y;
    int hull_count = end - start + 1;
    
    float4 diff = (float4)(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < hull_count; i++)
    {
        int n = start + i;
        float4 hull = hulls[n];
        int2 hull_flag = hull_flags[n];
        int4 element_table = element_tables[n];
        bool no_bones = (hull_flag.x & 0x08) !=0;

        if (!no_bones)
        {
            float2 center_a = calculate_centroid(points, element_table);
            float2 diffa = center_a - hull.xy;
            diff.x += diffa.x;
            diff.y += diffa.y;
            diff.z -= diffa.x;
            diff.w -= diffa.y;
        }
    }

    armature.x += diff.x;
    armature.y += diff.y;
    // armature.z += diff.x;
    // armature.w += diff.y;
    armatures[gid] = armature;
}