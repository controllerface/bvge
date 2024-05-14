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
__kernel void integrate(__global float2 *hulls,
                        __global float2 *hull_scales,
                        __global int2 *hull_point_tables,
                        __global float2 *armature_accel,
                        __global float2 *hull_rotations,
                        __global float4 *points,
                        __global ushort *point_hit_counts,
                        __global int *point_flags,
                        __global float4 *bounds,
                        __global int4 *bounds_index_data,
                        __global int2 *bounds_bank_data,
                        __global int *hull_flags,
                        __global int *hull_armature_ids,
                        __global float *anti_gravity,
                        __global float *args)
{
    int current_hull = get_global_id(0);

    float dt = args[0];
    float dt_2 = pown(dt, 2);
    float x_spacing = args[1];
    float y_spacing = args[2];
    float x_origin = args[3];
    float y_origin = args[4];
    float width = args[5];
    float height = args[6];
    int x_subdivisions = (int) args[7];
    int y_subdivisions = (int) args[8];
    float2 gravity = (float2)(args[9], args[10]);
    float damping = args[11];

    float inner_x_origin = args[12];
    float inner_y_origin = args[13];
    float inner_width = args[14];
    float inner_height = args[15];
    
    // get hull from array
    float2 hull = hulls[current_hull];
    float2 hull_scale = hull_scales[current_hull];
    int2 point_table = hull_point_tables[current_hull];
    int hull_1_flags = hull_flags[current_hull];
    int hull_armature_id = hull_armature_ids[current_hull];
    float2 acc = armature_accel[hull_armature_id];
    float2 rotation = hull_rotations[current_hull];
    float4 bounding_box = bounds[current_hull];
    int4 bounds_index = bounds_index_data[current_hull];
    int2 bounds_bank = bounds_bank_data[current_hull];

    // get start/end vertex indices
    int start = point_table.x;
    int end   = point_table.y;

    bool is_cursor = (hull_1_flags & IS_CURSOR) !=0;
    bool is_static = (hull_1_flags & IS_STATIC) !=0;
    bool is_circle = (hull_1_flags & IS_CIRCLE) !=0;
    bool no_bones = (hull_1_flags & NO_BONES) !=0;
    bool in_liquid = (hull_1_flags & IN_LIQUID) !=0;
    bool is_liquid = (hull_1_flags & IS_LIQUID) !=0;
    bool touch_alike = (hull_1_flags & TOUCH_ALIKE) !=0;
    bool out_of_bounds = (hull_1_flags & OUT_OF_BOUNDS) !=0;
    bool in_perimiter = (hull_1_flags & IN_PERIMETER) !=0;

    // wipe all ephemeral flags
    hull_1_flags &= ~OUT_OF_BOUNDS;
    hull_1_flags &= ~IN_PERIMETER;
    hull_1_flags &= ~IN_LIQUID;
    hull_1_flags &= ~TOUCH_ALIKE;
    hull_1_flags &= ~CURSOR_OVER;

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


        if (!is_static && !out_of_bounds && !is_cursor)
        {
            int _point_flags = point_flags[i];
            bool flow_left = (_point_flags & FLOW_LEFT) != 0;
            
            int hit_count = point_hit_counts[i];
            bool high_density = hit_count >= HIT_LOW_MID_THRESHOLD;
            _point_flags = high_density 
                ? _point_flags | HIGH_DENSITY 
                : _point_flags & ~HIGH_DENSITY;
            point_flags[i] =  _point_flags;

            // subtract prv from pos to get the difference this frame
            float2 diff = pos - prv;

            float g_x = touch_alike ? 0.01f : 0.01;
            float g_y = touch_alike ? 0.00f : 0.01;

 //float2 w_acc = (float2)(0.0f, 0.0f);
            float2 w_acc = (is_liquid)// & !touch_alike)
                ? flow_left
                    ? (float2)(-gravity.y * g_x, gravity.y * g_y)
                    : (float2)(gravity.y * g_x, gravity.y * g_y)
                : (float2)(0.0f, 0.0f);

            w_acc *= dt_2;

            diff = w_acc + acc + i_acc + diff;

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

        // update center sum
        x_sum += pos.x;
        y_sum += pos.y;

        // update min/max values for bounding box
        if (pos.x > max_x || !max_x_set)
        {
            max_x = pos.x;
            max_x_set = true;
        }
        if (pos.x < min_x || !min_x_set)
        {
            min_x = pos.x;
            min_x_set = true;
        }
        if (pos.y > max_y || !max_y_set)
        {
            max_y = pos.y;
            max_y_set = true;
        }
        if (pos.y < min_y || !min_y_set)
        {
            min_y = pos.y;
            min_y_set = true;
        }

        // store updated point in result buffer
        points[i] = point;
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
    bounding_box.x = min_x;
    bounding_box.y = min_y;
    bounding_box.z = fabs(max_x - min_x);
    bounding_box.w = fabs(max_y - min_y);

    // calculate spatial index boundary
    int2 keys[4];

    keys[0] = get_key_for_point(bounding_box.s0, bounding_box.s1, 
        x_spacing, y_spacing,
        x_origin, y_origin,
        width, height,
        x_subdivisions, y_subdivisions);

    keys[1] = get_key_for_point(max_x, bounding_box.s1, 
        x_spacing, y_spacing,
        x_origin, y_origin,
        width, height,
        x_subdivisions, y_subdivisions);

    keys[2] = get_key_for_point(max_x, max_y, 
        x_spacing, y_spacing,
        x_origin, y_origin,
        width, height,
        x_subdivisions, y_subdivisions);

    keys[3] = get_key_for_point(bounding_box.s0, max_y, 
        x_spacing, y_spacing,
        x_origin, y_origin,
        width, height,
        x_subdivisions, y_subdivisions);

    int4 k = getExtents(keys);
    bounds_index = k;
    bool in_bounds = is_in_bounds(bounding_box, x_origin, y_origin, width, height);

    if (in_bounds)
    {
        // calculate spatial index key bank size
        int x_count = (k.y - k.x) + 1;
        int y_count = (k.w - k.z) + 1;
        int count = x_count * y_count;
        int size = count * 2;
        bounds_bank.y = size;

        bool _in_perimiter = !is_in_bounds(bounding_box, inner_x_origin, inner_y_origin, inner_width, inner_height);
        if (_in_perimiter)
        {
            hull_1_flags |= IN_PERIMETER;
        }
    }
    else
    {
        bounds_bank.y = 0;
        if (!is_static && !is_cursor)
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

__kernel void integrate_armatures(__global float4 *armatures,
                                  __global int *armature_flags,
                                  __global int *armature_root_hulls,
                                  __global float2 *armature_accel,
                                  __global int *hull_flags,
                                  __global float *args)
{
    int current_armature = get_global_id(0);

    float dt = args[0];
    float2 gravity = (float2)(args[9], args[10]);
    float damping = args[11];

    float4 armature = armatures[current_armature];
    int _armature_flags = armature_flags[current_armature];
    bool is_wet = (_armature_flags & IS_WET) != 0;
    int root_hull = armature_root_hulls[current_armature];
    float2 acc = armature_accel[current_armature];
    int root_hull_flags = hull_flags[root_hull];

    bool is_static = (root_hull_flags & IS_STATIC) !=0;
    bool no_bones = (root_hull_flags & NO_BONES) !=0;

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

    float2 pos = armature.xy;
    float2 prv = armature.zw;

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
        armature.xy = pos;
        armature.zw = prv;
    }

    armatures[current_armature] = armature;
}
