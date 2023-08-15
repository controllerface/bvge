#define MINIMUM_DIFF 0.005

/**
Performs the integration step of a physics loop, generally this is the first stage
in a process that updates all the tracked vertices each frame.
Some meta-data about the hulls that are updated is stored within
them before this kernel completes. 
 */
__kernel void integrate(
    __global float4 *hulls,
    __global float4 *armatures,
    __global int *armature_flags,
    __global int4 *element_tables,
    __global float2 *armature_accel,
    __global float2 *hull_rotations,
    __global float4 *points,
    __global float4 *bounds,
    __global int4 *bounds_index_data,
    __global int2 *bounds_bank_data,
    __global int2 *hull_flags,
    __global float *args)
{
    int gid = get_global_id(0);

    float dt = args[0];
    float x_spacing = args[1];
    float y_spacing = args[2];
    float x_origin = args[3];
    float y_origin = args[4];
    float width = args[5];
    float height = args[6];
    int x_subdivisions = (int) args[7];
    int y_subdivisions = (int) args[8];
    float2 gravity;
    gravity.x = args[9];
    gravity.y = args[10];
    float friction = args[11];
    
    // get hull from array
    float4 hull = hulls[gid];
    int4 element_table = element_tables[gid];
    int2 hull_1_flags = hull_flags[gid];
    float4 armature = armatures[hull_1_flags.y];
    int armature_flag = armature_flags[hull_1_flags.y];
    float2 acceleration = armature_accel[hull_1_flags.y];
    float2 rotation = hull_rotations[gid];
    float4 bounding_box = bounds[gid];
    int4 bounds_index = bounds_index_data[gid];
    int2 bounds_bank = bounds_bank_data[gid];

    // get start/end vertex indices
    int start = element_table.x;
    int end   = element_table.y;

    // todo: instead of punting on these, we can maybe update differently and tag the hull
    //  or something, so it can be handled differently for collisions as well.
    // if (!is_in_bounds(bounding_box, x_origin, y_origin, width, height))
    // {
    //     acceleration.x = 0;
    //     acceleration.y = 0;
    //     armature_accel[gid] = acceleration;

    //     bounds_bank.y = 0;
    //     bounds_bank_data[gid] = bounds_bank;
    //     return;
    // }

   	// get acc value and multiply by the timestep do get the displacement vector
   	float2 acc;
    acc.x = acceleration.x;
    acc.y = acceleration.y;
    bool is_static = (hull_1_flags.x & 0x01) !=0;
    bool is_circle = (hull_1_flags.x & 0x02) !=0;

    if (!is_static)
    {
        acc.x += gravity.x;
        acc.y += gravity.y;
    }
   	acc.x = acc.x * (dt * dt);
   	acc.y = acc.y * (dt * dt);

    // only update the aramture during the update of the root hull, otherwise movement would be magnified 
    if (armature_flag == gid)
    {

        float2 pos = armature.xy;
        float2 prv = armature.zw;

        if (!is_static)
        {
            // subtract prv from pos to get the difference this frame
            float2 diff = pos - prv;
            diff = acc + diff;

            // add friction component todo: take this in as an argument, gravity too
            diff.x *= friction;
            diff.y *= friction;
            
            // set the prv to current pos
            prv.x = pos.x;
            prv.y = pos.y;

            if (diff.x < MINIMUM_DIFF && diff.x > -MINIMUM_DIFF)
            {
                diff.x = 0.0f;
            }

            if (diff.y < MINIMUM_DIFF && diff.y > -MINIMUM_DIFF)
            {
                diff.y = 0.0f;
            }

            // update pos
            pos = pos + diff;

            // finally, update the pos and prv in the object
            armature.x = pos.x;
            armature.y = pos.y;
            armature.z = prv.x;
            armature.w = prv.y;
        }

                // todo: do full integration, this is not correct and just moves a little bit
        armatures[hull_1_flags.y] = armature;
    }

    // reset acceleration to zero for the next frame
    acceleration.x = 0;
    acceleration.y = 0;

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

    bool inv_r = gid % 2 == 0;

    for (int i = start; i <= end; i++)
    {
        // get this point
        float4 point = points[i];
        
        // todo: move this idea somewhere else, it isn't goignt o work here anymore
        // force rotate the point to keep the object upright
        // todo: this should scale based on gravity, with zero g being no ro restriction of rotation
        //point = rotate_point(point, (float2)hull.xy, -rotation.x * 10);

        // this was a very basic orbital motion test, worth saving for something else
        //float rot  = /*inv_r ? -.00001 :*/ .00001;
        //point = rotate_point(point, (float2)(0,0), rot);

        // get pos/prv vectors
        float2 pos = point.xy;
        float2 prv = point.zw;

        if (!is_static)
        {
            // subtract prv from pos to get the difference this frame
            float2 diff = pos - prv;
            diff = acc + diff;

            // add friction component todo: take this in as an argument, gravity too
            diff.x *= friction;
            diff.y *= friction;
            
            // set the prv to current pos
            prv.x = pos.x;
            prv.y = pos.y;

            if (diff.x < MINIMUM_DIFF && diff.x > -MINIMUM_DIFF)
            {
                diff.x = 0.0f;
            }

            if (diff.y < MINIMUM_DIFF && diff.y > -MINIMUM_DIFF)
            {
                diff.y = 0.0f;
            }

            // update pos
            pos = pos + diff;

            // finally, update the pos and prv in the object
            point.x = pos.x;
            point.y = pos.y;
            point.z = prv.x;
            point.w = prv.y;
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

    // calculate centroid // todo: account for circles
    hull.x = x_sum / point_count;
    hull.y = y_sum / point_count;

    // handle bounding boxes for circles
    if (is_circle)
    {
        min_x = hull.x - hull.w;
        max_x = hull.x + hull.w;
        min_y = hull.y - hull.w;
        max_y = hull.y + hull.w;
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

    if (!is_in_bounds(bounding_box, x_origin, y_origin, width, height))
    {
        bounds_bank.y = 0;
    }
    else
    {
        // calculate spatial index key bank size
        int x_count = (k.y - k.x) + 1;
        int y_count = (k.w - k.z) + 1;
        int count = x_count * y_count;
        int size = count * 2;
        bounds_bank.y = size;
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

    // store updated hull and bounds data in result buffers
    bounds[gid] = bounding_box;
    hulls[gid] = hull;
    armature_accel[gid] = acceleration;
    hull_rotations[gid] = rotation;
    bounds_index_data[gid] = bounds_index;
    bounds_bank_data[gid] = bounds_bank;
}
