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
                          __global int *entity_model_transforms,
                          __global int *entity_flags,
                          __global float4 *hulls,
                          __global float2 *hull_scales,
                          __global float *hull_frictions,
                          __global float *hull_restitutions,
                          __global int *hull_integrity,
                          __global int2 *hull_point_tables,
                          __global int2 *hull_edge_tables,
                          __global int *hull_entity_ids,
                          __global int *hull_flags,
                          __global int *point_flags,
                          __global float4 *points,
                          __global int2 *edges,
                          __global int *edge_flags,
                          __global float8 *reactions,
                          __global int *reaction_index,
                          __global int *point_reactions,
                          __global float *masses,
                          __global int *counter,
                          float dt,
                          int max_index)
{
    int gid = get_global_id(0);
    if (gid >= max_index) return;
    
    int2 current_pair = candidates[gid];
    int b1_id = current_pair.x;
    int b2_id = current_pair.y;
    int hull_1_flags = hull_flags[b1_id];
    int hull_2_flags = hull_flags[b2_id];
    
    bool b1_is_circle = (hull_1_flags & IS_CIRCLE) !=0;
    bool b2_is_circle = (hull_2_flags & IS_CIRCLE) !=0;

    bool b1_is_block = (hull_1_flags & IS_BLOCK) !=0;
    bool b2_is_block = (hull_2_flags & IS_BLOCK) !=0;

    bool b1_is_polygon = (hull_1_flags & IS_POLYGON) !=0;
    bool b2_is_polygon = (hull_2_flags & IS_POLYGON) !=0;

    bool b1_is_sensor = (hull_1_flags & ENTITY_SENSOR) !=0;
    bool b2_is_sensor = (hull_2_flags & ENTITY_SENSOR) !=0;

    int c_id = b1_is_circle ? b1_id : b2_id;
    int p_id = b1_is_circle ? b2_id : b1_id;

    int e_id = b1_is_sensor ? b1_id : b2_id;
    int s_id = b1_is_sensor ? b2_id : b1_id;

    if (b1_is_block && b2_is_block) 
    {
        block_collision(b1_id, b2_id, 
            hulls,
            hull_frictions,
            hull_restitutions,
            hull_entity_ids,
            hull_flags,
            hull_point_tables,
            hull_edge_tables,
            point_flags,
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
    else if (b1_is_polygon && b2_is_polygon && !b1_is_sensor & !b2_is_sensor) 
    {
        polygon_collision(b1_id, b2_id, 
            entity_flags,
            hulls,
            hull_frictions,
            hull_restitutions,
            hull_integrity,
            hull_entity_ids,
            hull_flags,
            hull_point_tables,
            hull_edge_tables,
            point_flags,
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
            hull_scales,
            hull_frictions,
            hull_restitutions,
            hull_entity_ids,
            hull_flags,
            hull_point_tables,
            points,
            point_flags,
            reactions,
            reaction_index,
            point_reactions,
            masses,
            counter,
            dt); 
    }
    else if ((b1_is_sensor || b2_is_sensor) && !b1_is_circle && !b2_is_circle)
    {
        polygon_sensor_collision(s_id, e_id, 
                              hulls,
                              points,
                              edges,
                              hull_point_tables,
                              hull_edge_tables,
                              edge_flags,
                              hull_flags,
                              counter,
                              reactions,
                              reaction_index,
                              point_reactions,
                              dt);
    }
    else
    {
        polygon_circle_collision(p_id, c_id, 
            entity_model_transforms,
            entity_flags,
            hulls,
            hull_scales,
            hull_frictions,
            hull_restitutions,
            hull_integrity,
            hull_entity_ids,
            hull_flags,
            hull_point_tables,
            hull_edge_tables,
            point_flags,
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
                             __global int *point_offsets,
                             int max_index)
{
    int gid = get_global_id(0);
    if (gid >= max_index) return;

    float8 reaction = reactions_in[gid];
    int index = reaction_index[gid];
    int reaction_offset = point_offsets[index];
    int local_offset = atomic_inc(&point_reactions[index]);
    int next = reaction_offset + local_offset;
    reactions_out[next] = reaction;
}


inline float2 scale_reaction(float2 reaction, float dot_a, float dot_b)
{
    float2 norm = fast_normalize(reaction);
    float mag = fast_length(reaction);
    float scale = 1 - native_divide(dot_a, (dot_a + fabs(dot_b)));
    return norm * mag * scale;
}

/**
Applies reactions to points by summing all the reactions serially, and then applying the composite 
reaction to the point. 
 */
__kernel void apply_reactions(__global float8 *reactions,
                              __global float4 *points,
                              __global float *anti_gravity,
                              __global int *point_flags,
                              __global short *point_hit_counts,
                              __global int *point_reactions,
                              __global int *point_offsets,
                              __global int *point_hull_indices,
                              __global int *hull_flags,
                              int max_point)
{
    // todo: actual gravity vector should be provided, when it can change this should also be changable
    //  right now it is a static direction. note that magnitude of gravity is not important, only direction
    float2 g = (float2)(0.0f, -1.0f);

    int current_point = get_global_id(0);
    if (current_point >= max_point) return;

    int reaction_count = point_reactions[current_point];
    int flags = point_flags[current_point];
    short hit_count = point_hit_counts[current_point];
    
    int h_index = point_hull_indices[current_point];
    int h_flags = hull_flags[h_index];
    bool is_static = (h_flags & IS_STATIC) != 0;
    bool is_liquid = (h_flags & IS_LIQUID) != 0;

    // exit on non-reactive points
    if (reaction_count == 0) 
    {
        if (!is_static)
        {
            if(!is_liquid)
            {
                hit_count = hit_count == 0 
                    ? 0
                    : hit_count <= 3 
                    ? hit_count
                    : hit_count <= HIT_LOW_THRESHOLD 
                    ? hit_count - 3 
                    : hit_count <= HIT_LOW_MID_THRESHOLD
                    ? hit_count - 2
                    : hit_count <= HIT_MID_THRESHOLD 
                    ? hit_count - 1 
                    : hit_count <= HIT_HIGH_MID_THRESHOLD 
                    ? hit_count - 0 
                    : hit_count - 0;
            }
            else
            {
                hit_count = hit_count == 0 
                    ? 0
                    : hit_count <= HIT_LOW_THRESHOLD 
                    ? hit_count - 1 
                    : hit_count <= HIT_LOW_MID_THRESHOLD
                    ? hit_count - 2
                    : hit_count <= HIT_MID_THRESHOLD 
                    ? hit_count - 3 
                    : hit_count <= HIT_HIGH_MID_THRESHOLD 
                    ? hit_count - 4 
                    : hit_count - 5;
            }
        }

        point_hit_counts[current_point] = hit_count;
        point_flags[current_point] = flags;
        return;
    }
    
    hit_count = hit_count >= HIT_TOP_THRESHOLD 
        ? HIT_TOP_THRESHOLD 
        : hit_count + reaction_count;

    hit_count = hit_count > HIT_MID_THRESHOLD && reaction_count == 1
        ? hit_count - 2
        : hit_count;

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

    point.xy = (sign_a == sign_b) 
        ? point.xy + reaction.s45
        : point.xy + scale_reaction(reaction.s45, dot_a, dot_b);

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

    point.zw = (sign_a == sign_b) 
        ? point.zw + reaction.s67
        : point.zw + scale_reaction(reaction.s67, dot_a, dot_b);

    // in addition to velocity preservation, to aid in stabiliy, a non-real force of anti-gravity
    // is modeled to assist in keeping objects from colliding in the direction of gravity. This
    // adjustment is subtle and does not overcome all rigid-body simulation errors, but helps
    // maintain stability with small numbers of stacked objects. 
    float2 heading = reaction.s23;
    float ag = is_liquid ? 0.0f : calculate_anti_gravity(g, heading);

    flags = ag > 0.0f 
        ? flags | HIT_FLOOR
        : flags;

    // todo: anti-grav can take into account slope of colliding edge, if any. right now, only relative object direction is considered.

    // if anti-gravity would be negative, it means the heading is more in the direction of gravity 
    // than it is against it, so we clamp to 0.
    ag = ag <= 0.0f 
        ? 0.0f 
        : 1.0f;

    base_velocity = point.xy - point.zw;
    int _point_flags = point_flags[current_point];
    bool moving_left = base_velocity.x >= 0;

    _point_flags = moving_left
        ? _point_flags | FLOW_LEFT
        : _point_flags & ~FLOW_LEFT;
    
    point_flags[current_point] = _point_flags;
    anti_gravity[current_point] = ag;
    points[current_point] = point;
    point_flags[current_point] = flags;
    point_hit_counts[current_point] = hit_count;

    // It is important to reset the counts and offsets to 0 after reactions are handled.
    // These reactions are only valid once, for the current frame.
    point_reactions[current_point] = 0;
    point_offsets[current_point] = 0;
}

__kernel void move_hulls(__global float4 *hulls,
                         __global int2 *hull_point_tables,
                         __global float4 *points,
                         int max_hull)
{
    int current_hull = get_global_id(0);
    if (current_hull >= max_hull) return;
    int2 point_table = hull_point_tables[current_hull];
    float4 hull = hulls[current_hull];
    hull.zw = hull.xy;
    hull.xy = calculate_centroid(points, point_table);
    hulls[current_hull] = hull;
}

inline int2 consume_point_flags(__global int *point_flags,
                               __global short *point_hit_counts,
                               int2 point_table)
{
    int2 result = (int2)(0, 0);

    int start = point_table.x;
    int end   = point_table.y;
	int vert_count = end - start + 1;

    for (int i = 0; i < vert_count; i++)
    {
        int n = start + i;
        int flags = point_flags[n];
        short pc = point_hit_counts[n];
        result.x |= flags;
        result.y += (int)pc;

        // the floor flag must be "consumed" so it doesn't persist to the next tick
        flags &= ~HIT_FLOOR;
        
        point_flags[n] = flags;
    }

    return result;
}
__kernel void move_entities(__global float4 *hulls,
                            __global float4 *entities,
                            __global int *entity_flags,
                            __global short2 *entity_motion_states,
                            __global int2 *hull_tables,
                            __global int2 *hull_point_tables,
                            __global int *hull_integrity,
                            __global int *hull_flags,
                            __global int *point_flags,
                            __global short *point_hit_counts,
                            __global float4 *points,
                            float dt,
                            int max_entity)
{
    int current_entity = get_global_id(0);
    if (current_entity >= max_entity) return;
    float4 entity = entities[current_entity];
    int flags = entity_flags[current_entity];
    int2 hull_table = hull_tables[current_entity];
    int start = hull_table.x;
    int end = hull_table.y;
    int hull_count = end - start + 1;
    short2 motion_state = entity_motion_states[current_entity];

    int hull_flags_0 = hull_flags[start];
    bool is_block = (hull_flags_0 & IS_BLOCK) != 0;
    bool collectable = (flags & COLLECTABLE) != 0;
    bool jumping = (flags & JUMPING) != 0;

    int hull_0_integrity = hull_integrity[start];
    bool single_hull = hull_count == 1;

    float2 diff = (float2)(0.0f);
    int _point_flags = 0;
    int _hull_flags = 0;
    float2 last_center = (float2)(0.0f);
    bool had_bones = false;
    bool had_touch = false;
    int total_hits = 0;
    int has_sensor = false;
    float2 sensor_p1 = (float2)(0.0f);
    for (int i = 0; i < hull_count; i++)
    {
        int n = start + i;
        float4 hull = hulls[n];
        int hull_flag = hull_flags[n];
        int2 point_table = hull_point_tables[n];
        bool no_bones = (hull_flag & NO_BONES) !=0;
        bool is_foot = (hull_flag & IS_FOOT) !=0;
        bool is_sensor = (hull_flag & IS_SENSOR) !=0;
        bool e_is_sensor = (hull_flag & ENTITY_SENSOR) !=0;

        if (is_sensor)
        {
            bool istouch = (hull_flag & SENSOR_HIT) !=0;
            had_touch = istouch 
                ? true 
                : had_touch;
            if (!e_is_sensor) continue;
            //has_sensor = true;
            //sensor_p1 = points[point_table.x].xy;
        } 

        _hull_flags |= hull_flag;

        if (!no_bones) had_bones = true;

        last_center = hull.xy;
        float2 diffa = last_center - hull.zw;
        diff += diffa;
        int2 xa = consume_point_flags(point_flags, point_hit_counts, point_table);
        total_hits += xa.y;
        _point_flags = is_foot || is_block
            ? _point_flags | xa.x
            : _point_flags;
    }

    bool hit_floor = (_point_flags & HIT_FLOOR) !=0;
    bool hit_water = (_hull_flags & IN_LIQUID) !=0;
    bool touch_alike = (_hull_flags & TOUCH_ALIKE) !=0;

    int block_check = HIT_TOP_THRESHOLD;

    // bool go_static = hit_floor  
    //     && !hit_water 
    //     && !collectable 
    //     && is_block 
    //     && total_hits >= block_check;

    bool destroy = single_hull && hull_0_integrity <= 0;
    
    // hull_flags_0 = go_static 
    //     ? hull_flags_0 | IS_STATIC
    //     : hull_flags_0;
    
    entity.w = !jumping && hit_floor && had_touch 
        ? entity.y 
        : entity.w;

    float2 initial_tail = entity.zw;
    float initial_dist = fast_distance(entity.xy, entity.zw);

    entity.xy = had_bones 
        ? entity.xy + diff
        : last_center;

    entity.xy = has_sensor 
        ? sensor_p1 
        : entity.xy;

    float2 adjusted_offset = entity.xy - initial_tail;
    float new_len = fast_length(adjusted_offset);

    adjusted_offset = new_len == 0.0f 
        ? adjusted_offset 
        : native_divide(adjusted_offset, new_len);

    entity.zw = entity.xy - initial_dist * adjusted_offset;

    flags = had_touch 
        ? flags | CAN_JUMP
        : flags & ~CAN_JUMP;

    flags = hit_water
        ? flags | IS_WET
        : flags & ~IS_WET;

    flags = destroy
        ? flags | BROKEN
        : flags & ~BROKEN;

    float threshold = 10.0f;
    float2 vel = (entity.xy - entity.zw) / dt;

    motion_state.x = (vel.y < -threshold) 
        ? motion_state.x + 1 
        : 0;

    motion_state.y = (vel.y > threshold) 
        ? motion_state.y + 1 
        : 0;

    motion_state.x = motion_state.x > 1000 
        ? 1000 
        : motion_state.x;

    motion_state.y = motion_state.y > 1000 
        ? 1000 
        : motion_state.y;

    entity_motion_states[current_entity] = motion_state;
    hull_flags[start] = hull_flags_0;
    entities[current_entity] = entity;
    entity_flags[current_entity] = flags;
}
