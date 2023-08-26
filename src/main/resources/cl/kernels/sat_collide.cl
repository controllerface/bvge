/**
Performs collision detection using separating axis theorem, and then applys a reaction
for objects when they are found to be colliding. Reactions detemine one "edge" polygon 
and one "vertex" polygon. The vertex polygon has a single vertex adjusted as a reaction. 
The edge object has two vertices adjusted and the adjustments are in oppostie directions, 
which will naturally apply some degree of rotation to the object.
 todo: add circles, currently assumes polygons 
 */
__kernel void sat_collide(__global int2 *candidates,
                          __global float4 *hulls,
                          __global float4 *armatures,
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
    float4 b1_hull = hulls[b1_id];
    float4 b2_hull = hulls[b2_id];
    int2 hull_1_flags = hull_flags[b1_id];
    int2 hull_2_flags = hull_flags[b2_id];
    bool b1s = (hull_1_flags.x & 0x01) !=0;
    bool b2s = (hull_2_flags.x & 0x01) !=0;
    float4 b1_armature = armatures[hull_1_flags.y];
    float4 b2_armature = armatures[hull_2_flags.y];
    int4 hull_1_table = element_tables[b1_id];
    int4 hull_2_table = element_tables[b2_id];
    
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


    bool b1_no_bones = (hull_1_flags.x & 0x08) !=0;
    bool b2_no_bones = (hull_2_flags.x & 0x08) !=0;

    // todo: this needs to be moved to a separate kernel, later in the physics loop
    if (!b1_no_bones)
    {
        float2 center_a = calculate_centroid(points, hull_1_table);
        float2 diffa = center_a - b1_hull.xy;
        b1_armature.x += diffa.x;
        b1_armature.y += diffa.y;
        b1_armature.z -= diffa.x;
        b1_armature.w -= diffa.y;
        // b1_armature.z = b1_armature.x -= diffa.x;
        // b1_armature.w = b1_armature.y -= diffa.y;
         armatures[hull_1_flags.y] = b1_armature;
    }

    if (!b2_no_bones)
    {
        float2 center_b = calculate_centroid(points, hull_2_table);
        float2 diffb = center_b - b2_hull.xy;
        b2_armature.x += diffb.x;
        b2_armature.y += diffb.y;
        b2_armature.z -= diffb.x;
        b2_armature.w -= diffb.y;
        // b2_armature.z = b2_armature.x -= diffb.x;
        // b2_armature.w = b2_armature.y -= diffb.y;
        armatures[hull_2_flags.y] = b2_armature;
    }
}

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

    barrier(CLK_GLOBAL_MEM_FENCE);

    local_offset;

    int next = reaction_offset + local_offset;

    //printf("debug offset local=%d global=%d", next, reaction_offset);

    reactions[next] = reaction;
    reaction_index[next] = index;
}

__kernel void apply_reactions(__global float2 *reactions,
                              __global float4 *points,
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
    for (int i = 0; i < reaction_count; i++)
    {
        int idx = i + offset;
        float2 reaction_i = reactions[idx];
        reaction += reaction_i;
    }

    point_reactions[gid] = 0;
    point_offsets[gid] = 0;

    point.xy += reaction;

    float2 e1_diff_2 = point.xy - e1_p;
    float new_len_e1 = length(e1_diff_2);

    if (new_len_e1 != 0.0)
    {
        e1_diff_2 /= new_len_e1;
        point.zw = point.xy - e1_dist * e1_diff_2;
    }

    points[gid] = point;
}

__kernel void move_armatures(__global float4 *hulls,
                             __global float4 *armatures,
                             __global int4 *element_tables,
                             __global int2 *hull_flags,
                             __global float4 *points)
{
    int gid = get_global_id(0);
    float4 b1_hull = hulls[gid];
    int2 hull_1_flags = hull_flags[gid];
    int4 hull_1_table = element_tables[gid];
    float4 b1_armature = armatures[hull_1_flags.y];

    bool b1_no_bones = (hull_1_flags.x & 0x08) !=0;

    // todo: this needs to be moved to a separate kernel, later in the physics loop
    if (!b1_no_bones)
    {
        float2 center_a = calculate_centroid(points, hull_1_table);
        float2 diffa = center_a - b1_hull.xy;
        b1_armature.x += diffa.x;
        b1_armature.y += diffa.y;
        b1_armature.z -= diffa.x;
        b1_armature.w -= diffa.y;
        // b1_armature.z = b1_armature.x -= diffa.x;
        // b1_armature.w = b1_armature.y -= diffa.y;
        armatures[hull_1_flags.y] = b1_armature;
    }

}