#define MINIMUM_DIFF 0.005

/**
Performs the integration step of a physics loop, generally this is the first stage
in a process that updates all the tracked vertices each frame.
Some meta-data about the bodies that are updated is stored within
them before this kernel completes. 
 */

// todo: convert to: 
//  - float 4, transform
//  - float 2, accel
//  - int 4, element table
//  - int 2, key bank
//  - float 4, extents
//  - int 4, aabb index


__kernel void integrate(
    __global float16 *bodies,
    __global float2 *body_accel,
    __global float4 *points,
    __global float16 *bounds,
    __global int4 *bounds_index_data,
    __global int2 *bounds_bank_data,
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
    
    // get body from array
    float16 body = bodies[gid];
    float2 acceleration = body_accel[gid];
    float16 bounding_box = bounds[gid];
    int4 bounds_index = bounds_index_data[gid];
    int2 bounds_bank = bounds_bank_data[gid];


    // get start/end vertex indices
    int start = (int)body.s7;
    int end   = (int)body.s8;

    // todo: instead of punting on these, we can maybe update differently and tag the body
    //  or something, so it can be handled differently for collisions as well.
    if (!is_in_bounds(bounding_box, x_origin, y_origin, width, height))
    {
        acceleration.x = 0;
        acceleration.y = 0;
        body_accel[gid] = acceleration;

        bounds_bank.y = 0;
        bounds_bank_data[gid] = bounds_bank;
        return;
    }

   	// get acc value and multiply by the timestep do get the displacement vector
   	float2 acc;
    acc.x = acceleration.x;
    acc.y = acceleration.y;
    bool b1s = (body.s6 && 0x01) !=0;
    
    if (!b1s)
    {
        acc.x += gravity.x;
        acc.y += gravity.y;
    }
   	acc.x = acc.x * (dt * dt);
   	acc.y = acc.y * (dt * dt);

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

    for (int i = start; i <= end; i++)
    {
        // get this point
        float4 point = points[i];

        // get pos/prv vectors
        float2 pos = point.xy;
        float2 prv = point.zw;

        if (!b1s)
        {
            // subtract prv from pos to get the difference this frame
            float2 diff = pos - prv;
            diff = acc + diff;

            // add friction component todo: take this in as an argument, gravity too
            diff.x *= friction;
            diff.y *= friction;
            
            if (diff.x < MINIMUM_DIFF && diff.x > -MINIMUM_DIFF)
            {
                diff.x = 0.0f;
            }

            if (diff.y < MINIMUM_DIFF && diff.y > -MINIMUM_DIFF)
            {
                diff.y = 0.0f;
            }

            // set the prv to current pos
            prv.x = pos.x;
            prv.y = pos.y;

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

    // calculate centroid
    body.s0 = x_sum / point_count;
    body.s1 = y_sum / point_count;

    // calculate bounding box
    bounding_box.s0 = min_x;
    bounding_box.s1 = min_y;
    bounding_box.s2 = fabs(max_x - min_x);
    bounding_box.s3 = fabs(max_y - min_y);

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

    // store updated body and bounds data in result buffers
    bounds[gid] = bounding_box;
    bodies[gid] = body;
    body_accel[gid] = acceleration;
    bounds_index_data[gid] = bounds_index;
    bounds_bank_data[gid] = bounds_bank;
}
