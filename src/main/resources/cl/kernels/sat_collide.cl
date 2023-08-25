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
        polygon_collision(b1_id, b2_id, hulls, hull_flags, element_tables, points, edges, counter); 
    }
    else if (b1_is_circle && b2_is_circle) 
    {
        circle_collision(b1_id, b2_id, hulls, element_tables, points, counter); 
    }
    else 
    {
        polygon_circle_collision(p_id, c_id, hulls, hull_flags, element_tables, points, edges, counter); 
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
