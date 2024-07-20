inline float2 generate_counter_vector(float2 vector, float counter_scale)
{
    float2 neg;
    neg.x = -vector.x * (counter_scale);  
    neg.y = -vector.y * (counter_scale); 
    return neg; 
}

/**
Performs the integration step of a physics loop, generally this is the first stage
in a process that updates all the tracked vertices each frame.
Some meta-data about the hulls that are updated is stored within
them before this kernel completes. 
 */
__kernel void integrate(__global int2 *hull_point_tables,
                        __global float2 *entity_accel,
                        __global float4 *points,
                        __global short *point_hit_counts,
                        __global int *point_flags,
                        __global int *hull_flags,
                        __global int *hull_entity_ids,
                        __global float *anti_gravity,
                        __global float *args, 
                        int max_hull)
{
    int current_hull = get_global_id(0);

    if (current_hull >= max_hull) return;

    float dt             = args[0];
    float2 gravity       = (float2)(args[1], args[2]);
    float damping        = args[3];

    
    float dt_2 = pown(dt, 2);
    
    // get hull from array
    int2 point_table = hull_point_tables[current_hull];
    int hull_1_flags = hull_flags[current_hull];
    int hull_entity_id = hull_entity_ids[current_hull];
    float2 acc = entity_accel[hull_entity_id];

    // get start/end vertex indices
    int start = point_table.x;
    int end   = point_table.y;

    bool is_cursor     = (hull_1_flags & IS_CURSOR) !=0;
    bool is_ghost      = (hull_1_flags & GHOST_HULL) !=0;
    bool is_static     = (hull_1_flags & IS_STATIC) !=0;
    bool is_circle     = (hull_1_flags & IS_CIRCLE) !=0;
    bool no_bones      = (hull_1_flags & NO_BONES) !=0;
    bool in_liquid     = (hull_1_flags & IN_LIQUID) !=0;
    bool is_liquid     = (hull_1_flags & IS_LIQUID) !=0;
    bool touch_alike   = (hull_1_flags & TOUCH_ALIKE) !=0;
    bool out_of_bounds = (hull_1_flags & OUT_OF_BOUNDS) !=0;
    bool in_perimiter  = (hull_1_flags & IN_PERIMETER) !=0;

    // wipe all ephemeral flags
    hull_1_flags &= ~OUT_OF_BOUNDS;
    hull_1_flags &= ~IN_PERIMETER;
    hull_1_flags &= ~IN_LIQUID;
    hull_1_flags &= ~TOUCH_ALIKE;
    hull_1_flags &= ~CURSOR_OVER;
    hull_1_flags &= ~IN_RANGE;
    hull_1_flags &= ~CURSOR_HIT;

    gravity = in_liquid
        ? gravity * 1.5f
        : gravity;

    gravity = in_perimiter
        ? (float2)(0.0f)
        : gravity;

   	// get acc value and multiply by the timestep do get the displacement vector
    acc = is_static 
        ? acc
        : acc + gravity;

   	acc *= dt_2;

	// calculate the number of vertices, used later for centroid calculation
	int point_count = end - start + 1;

	// track the center index for the centroid calc at the end as well as the min/max for bounding box
	float x_sum = 0;
	float y_sum = 0;

	bool min_x_set = false;
	bool max_x_set = false;
	bool min_y_set = false;
	bool max_y_set = false;

	float min_x = FLT_MAX;
	float max_x = FLT_MIN;
	float min_y = FLT_MAX;
	float max_y = FLT_MIN;

    float anti_grav_scale = 0;
    for (int i = start; i <= end; i++)
    {
        float point_scale = anti_gravity[i];
        anti_gravity[i] = 0;
        anti_grav_scale = max(point_scale, anti_grav_scale);
    }

    float2 anti_grav = generate_counter_vector(gravity, anti_grav_scale);
    float2 i_acc = anti_grav * dt_2;

    float y_damping = in_liquid
        ? .920f
        : 1.0f;

    for (int i = start; i <= end; i++)
    {

        // float anti_grav_scale = 0;
        // //for (int i = start; i <= end; i++)
        // //{
        //     float point_scale = anti_gravity[i];
        //     anti_gravity[i] = 0;
        //     anti_grav_scale = max(point_scale, anti_grav_scale);
        // //}

        // float2 anti_grav = generate_counter_vector(gravity, anti_grav_scale);
        // float2 i_acc = anti_grav * dt_2;

        // get this point
        float4 point = points[i];
        
        // get pos/prv vectors
        float2 pos = point.xy;
        float2 prv = in_perimiter || out_of_bounds ? point.xy : point.zw;

        float x_threshold = is_liquid ? 1.0f : 5.0f;
        float y_threshold = is_liquid ? 0.25f : 1.0f;
        
        float2 vel = (pos - prv) ;/// dt;
        bool s_x = fabs(vel.x) > x_threshold;
        bool s_y = in_liquid && fabs(vel.y) > y_threshold;

        float sign_x = vel.x < 0 ? -1 : 1;
        // float sign_y = vel.y < 0 ? -1 : 1;


        prv.x = s_x ? pos.x - sign_x * x_threshold : prv.x;
        // prv.y = s_y ? pos.y - sign_y * y_threshold : prv.y;

        if (is_cursor || is_ghost)
        {
            point.zw = point.xy;
            points[i] = point;
        }


        if (!is_static && !out_of_bounds && !is_cursor && !is_ghost)
        {
            int _point_flags = point_flags[i];
            bool flow_left = (_point_flags & FLOW_LEFT) != 0;
            
            int hit_count = point_hit_counts[i];
            bool high_density = hit_count >= HIT_LOW_MID_THRESHOLD;
            bool max_density = hit_count >= HIT_TOP_THRESHOLD;
            _point_flags = high_density 
                ? _point_flags | HIGH_DENSITY 
                : _point_flags & ~HIGH_DENSITY;
            point_flags[i] =  _point_flags;

            // subtract prv from pos to get the difference this frame
            float2 diff = pos - prv;

            float g_x = touch_alike 
                ? high_density 
                    ? 0.1f
                    : max_density 
                        ? 0.3f 
                        : 0.0f
                : 0.0;

         float g_y = touch_alike 
                ? high_density 
                    ? 0.1f
                    : max_density 
                        ? 0.3f 
                        : 0.0f
                : 0.0;

            //float g_y = touch_alike ? 0.02f : 0.0;

            float2 w_acc = (is_liquid)
                ? flow_left
                    ? (float2)(-gravity.y * g_x, gravity.y * g_y)
                    : (float2)(gravity.y * g_x, gravity.y * g_y)
                : (float2)(0.0f, 0.0f);

            w_acc *= dt_2;

            diff = w_acc + acc + i_acc + diff;

            if (is_liquid) 
            { 
                diff.x = diff.x * 1.003; 
                diff.y = diff.y * .992; 
            }

            // add damping component
            diff.x = is_liquid 
                ? diff.x
                : diff.x * min(y_damping, damping);
            diff.y = s_y 
                ? diff.y * y_damping 
                : diff.y;
            diff.y = diff.y > 0 
                ? diff.y * damping 
                : diff.y;

        
            // set the prv to current pos
            prv = pos;

            // update pos
            pos = pos + diff;

            // finally, update the pos and prv in the object
            point.xy = pos;
            point.zw = prv;
        }

        // store updated point in result buffer
        points[i] = point;
    }

    hull_flags[current_hull] = hull_1_flags;
}

__kernel void integrate_entities(__global float4 *entities,
                                  __global int *entity_flags,
                                  __global int *entity_root_hulls,
                                  __global float2 *entity_accel,
                                  __global int *hull_flags,
                                  __global float *args, 
                                  int max_entity)
{
    int current_entity = get_global_id(0);

    if (current_entity >= max_entity) return;

    float dt = args[0];
    float2 gravity = (float2)(args[1], args[2]);
    float damping = args[3];
    float sector_x = args[4];
    float sector_y = args[5];
    float sector_w = args[6];
    float sector_h = args[7];

    float4 entity = entities[current_entity];
    int _entity_flags = entity_flags[current_entity];
    bool is_wet = (_entity_flags & IS_WET) != 0;
    int root_hull = entity_root_hulls[current_entity];
    float2 acc = entity_accel[current_entity];
    int root_hull_flags = hull_flags[root_hull];

    bool is_static = (root_hull_flags & IS_STATIC) !=0;
    bool no_bones  = (root_hull_flags & NO_BONES) !=0;
    bool is_cursor = (root_hull_flags & IS_CURSOR) !=0;
    bool is_ghost  = (root_hull_flags & GHOST_HULL) !=0;

    //gravity = (float2)(0.0f);

    gravity = is_wet
        ? gravity * 0.4f
        : gravity;

    float y_damping = is_wet
        ? .985f
        : 1.0f;

    acc = is_static
        ? acc
        : acc + gravity;
   	acc *= (dt * dt);

    float2 pos = entity.xy;
    float2 prv = entity.zw;

    if (!is_static && !no_bones)
    {
        float2 diff = pos - prv;
        diff = acc + diff;
        diff.x *= damping;
        diff.y *= y_damping;
        diff.y = diff.y > 0
            ? diff.y * damping
            : diff.y;

        prv = pos;
        pos = pos + diff;
        entity.xy = pos;
        entity.zw = prv;
    }

    bool sector_in = is_cursor || is_ghost || is_point_in_bounds(pos, sector_x, sector_y, sector_w, sector_h);

    _entity_flags = !sector_in 
        ? _entity_flags | SECTOR_OUT
        : _entity_flags; 

    entities[current_entity] = entity;
    entity_flags[current_entity] = _entity_flags;
}

__kernel void calculate_hull_aabb(__global float4 *hulls,
                                  __global float2 *hull_scales,
                                  __global int2 *hull_point_tables,
                                  __global float2 *hull_rotations,
                                  __global float4 *points,
                                  __global float4 *bounds,
                                  __global int4 *bounds_index_data,
                                  __global int2 *bounds_bank_data,
                                  __global int *hull_flags,
                                  __global float *args, 
                                  int max_hull)
{
    int current_hull = get_global_id(0);

    if (current_hull >= max_hull) return;

    float x_spacing      = args[0];
    float y_spacing      = args[1];
    float x_origin       = args[2];
    float y_origin       = args[3];
    float width          = args[4];
    float height         = args[5];
    float inner_x_origin = args[6];
    float inner_y_origin = args[7];
    float inner_width    = args[8];
    float inner_height   = args[9];
        
    // get hull from array
    float4 hull = hulls[current_hull];
    float2 hull_scale = hull_scales[current_hull];
    int2 point_table = hull_point_tables[current_hull];
    int hull_1_flags = hull_flags[current_hull];
    float2 rotation = hull_rotations[current_hull];
    int4 bounds_index = bounds_index_data[current_hull];
    int2 bounds_bank = bounds_bank_data[current_hull];

    // get start/end vertex indices
    int start = point_table.x;
    int end   = point_table.y;

    bool is_cursor     = (hull_1_flags & IS_CURSOR) !=0;
    bool is_ghost      = (hull_1_flags & GHOST_HULL) !=0;
    bool is_circle     = (hull_1_flags & IS_CIRCLE) !=0;
    bool out_of_bounds = (hull_1_flags & OUT_OF_BOUNDS) !=0;

	// calculate the number of vertices, used later for centroid calculation
	int point_count = end - start + 1;

	// track the center index for the centroid calc at the end as well as the min/max for bounding box
	float x_sum = 0;
	float y_sum = 0;

	bool min_x_set = false;
	bool max_x_set = false;
	bool min_y_set = false;
	bool max_y_set = false;

	float min_x = FLT_MAX;
	float max_x = FLT_MIN;
	float min_y = FLT_MAX;
	float max_y = FLT_MIN;

    for (int i = start; i <= end; i++)
    {
        // get this point
        float4 point = points[i];
        
        // update center sum
        x_sum += point.x;
        y_sum += point.y;

        // update min/max values for bounding box
        if (point.x > max_x || !max_x_set)
        {
            max_x = point.x;
            max_x_set = true;
        }
        if (point.x < min_x || !min_x_set)
        {
            min_x = point.x;
            min_x_set = true;
        }
        if (point.y > max_y || !max_y_set)
        {
            max_y = point.y;
            max_y_set = true;
        }
        if (point.y < min_y || !min_y_set)
        {
            min_y = point.y;
            min_y_set = true;
        }

        // also include previous position data for the bounding box, to account for possible CCD correction
        if (point.z > max_x)
        {
            max_x = point.z;
        }
        if (point.z < min_x)
        {
            min_x = point.z;
        }
        if (point.w > max_y)
        {
            max_y = point.w;
        }
        if (point.w < min_y)
        {
            min_y = point.w;
        }
    }

    // calculate centroid
    hull.x = native_divide(x_sum, point_count);
    hull.y = native_divide(y_sum, point_count);

    // handle bounding boxes for circles
    if (is_circle)
    {
        min_x = hull.x - hull_scale.y;
        max_x = hull.x + hull_scale.y;
        min_y = hull.y - hull_scale.y;
        max_y = hull.y + hull_scale.y;
    }

    // calculate bounding box
    float4 bounding_box = (float4)(0.0f);
    bounding_box.x = min_x;
    bounding_box.y = min_y;
    bounding_box.z = fabs(max_x - min_x);
    bounding_box.w = fabs(max_y - min_y);

    // calculate spatial index boundary
    int2 keys[4];

    keys[0] = get_key_for_point(bounding_box.s0, bounding_box.s1, 
        x_spacing, y_spacing,
        x_origin, y_origin);

    keys[1] = get_key_for_point(max_x, bounding_box.s1, 
        x_spacing, y_spacing,
        x_origin, y_origin);

    keys[2] = get_key_for_point(max_x, max_y, 
        x_spacing, y_spacing,
        x_origin, y_origin);

    keys[3] = get_key_for_point(bounding_box.s0, max_y, 
        x_spacing, y_spacing,
        x_origin, y_origin);

    int4 k = getExtents(keys);
    bounds_index = k;
    bool in_bounds = is_box_in_bounds(bounding_box, x_origin, y_origin, width, height);

    if (in_bounds)
    {
        // calculate spatial index key bank size
        int x_count = (k.y - k.x) + 1;
        int y_count = (k.w - k.z) + 1;
        int count = x_count * y_count;
        int size = count * 2;
        bounds_bank.y = size;

        bool _in_perimiter = !is_box_in_bounds(bounding_box, inner_x_origin, inner_y_origin, inner_width, inner_height);
        if (_in_perimiter)
        {
            hull_1_flags |= IN_PERIMETER;
        }
    }
    else
    {
        bounds_bank.y = 0;
        if (!is_cursor && !is_ghost)
        {
            hull_1_flags |= OUT_OF_BOUNDS;
        }
    }

    float4 ref_point = points[start];
    
    // unit vector up from where we are now
    float4 l1 = (float4)(hull.x, hull.y, hull.x, hull.y + 1);
    
    // vector pointing at the reference point
    float4 l2 = (float4)(hull.x, hull.y, ref_point.x, ref_point.y);
        
    // determine the rotation of the hull
    float r_x = angle_between(l1, l2);
    
    // rotation.y is the reference angle taken at object creation when rotation is zero
    rotation.x = rotation.y - r_x;

    hull_flags[current_hull] = hull_1_flags;
    bounds[current_hull] = bounding_box;
    hulls[current_hull] = hull;
    hull_rotations[current_hull] = rotation;
    bounds_index_data[current_hull] = bounds_index;
    bounds_bank_data[current_hull] = bounds_bank;
}

__kernel void calculate_edge_aabb(__global int2 *edges,
                                  __global int *edge_flags,
                                  __global float4 *points,
                                  __global float4 *edge_aabb,
                                  __global int4 *edge_aabb_index,
                                  __global int2 *edge_aabb_key_table,
                                  __global float *args,
                                  int max_edge)
{
    int current_edge = get_global_id(0);

    if (current_edge >= max_edge) return;

    float x_spacing      = args[0];
    float y_spacing      = args[1];
    float x_origin       = args[2];
    float y_origin       = args[3];
    float width          = args[4];
    float height         = args[5];
    float inner_x_origin = args[6];
    float inner_y_origin = args[7];
    float inner_width    = args[8];
    float inner_height   = args[9];
        
    int2 edge = edges[current_edge];
    float4 point1 = points[edge.x];
    float4 point2 = points[edge.y];
    int4 bounds_index = edge_aabb_index[current_edge];
    int2 bounds_bank = edge_aabb_key_table[current_edge];

	float x_sum = 0;
	float y_sum = 0;

	bool min_x_set = false;
	bool max_x_set = false;
	bool min_y_set = false;
	bool max_y_set = false;

	float min_x = FLT_MAX;
	float max_x = FLT_MIN;
	float min_y = FLT_MAX;
	float max_y = FLT_MIN;

    // update min/max values for bounding box
    if (point1.x > max_x || !max_x_set)
    {
        max_x = point1.x;
        max_x_set = true;
    }
    if (point1.x < min_x || !min_x_set)
    {
        min_x = point1.x;
        min_x_set = true;
    }
    if (point1.y > max_y || !max_y_set)
    {
        max_y = point1.y;
        max_y_set = true;
    }
    if (point1.y < min_y || !min_y_set)
    {
        min_y = point1.y;
        min_y_set = true;
    }

    if (point2.x > max_x)
    {
        max_x = point2.x;
    }
    if (point2.x < min_x)
    {
        min_x = point2.x;
    }
    if (point2.y > max_y)
    {
        max_y = point2.y;
    }
    if (point2.y < min_y)
    {
        min_y = point2.y;
    }

    // also include previous position data for the bounding box, to account for possible CCD correction
    if (point1.z > max_x)
    {
        max_x = point1.z;
    }
    if (point1.z < min_x)
    {
        min_x = point1.z;
    }
    if (point1.w > max_y)
    {
        max_y = point1.w;
    }
    if (point1.w < min_y)
    {
        min_y = point1.w;
    }
    if (point2.z > max_x)
    {
        max_x = point2.z;
    }
    if (point2.z < min_x)
    {
        min_x = point2.z;
    }
    if (point2.w > max_y)
    {
        max_y = point2.w;
    }
    if (point2.w < min_y)
    {
        min_y = point2.w;
    }

	min_x -= .5f;
	max_x += .5f;
	min_y -= .5f;
	max_y += .5f;

    // calculate bounding box
    float4 bounding_box = (float4)(0.0f);
    bounding_box.x = min_x;
    bounding_box.y = min_y;
    bounding_box.z = fabs(max_x - min_x);
    bounding_box.w = fabs(max_y - min_y);

    // calculate spatial index boundary
    int2 keys[4];

    keys[0] = get_key_for_point(bounding_box.s0, bounding_box.s1, 
        x_spacing, y_spacing,
        x_origin, y_origin);

    keys[1] = get_key_for_point(max_x, bounding_box.s1, 
        x_spacing, y_spacing,
        x_origin, y_origin);

    keys[2] = get_key_for_point(max_x, max_y, 
        x_spacing, y_spacing,
        x_origin, y_origin);

    keys[3] = get_key_for_point(bounding_box.s0, max_y, 
        x_spacing, y_spacing,
        x_origin, y_origin);

    int4 k = getExtents(keys);
    bounds_index = k;

    bool interior =  edge_flags[current_edge] == 1;

    bool in_bounds = !interior && is_box_in_bounds(bounding_box, x_origin, y_origin, width, height);

    if (in_bounds)
    {
        // calculate spatial index key bank size
        int x_count = (k.y - k.x) + 1;
        int y_count = (k.w - k.z) + 1;
        int count = x_count * y_count;
        int size = count * 2;
        bounds_bank.y = size;
    }
    else
    {
        bounds_bank.y = 0;
    }

    edge_aabb[current_edge]           = bounding_box;
    edge_aabb_index[current_edge]     = bounds_index;
    edge_aabb_key_table[current_edge] = bounds_bank;
}
