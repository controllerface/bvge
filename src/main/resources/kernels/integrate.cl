__kernel void integrate(
    __global const float16 *bodies,
    __global const float4 *points,
    __global const float8 *bounds,
    __global float16 *r_bodies,
    __global float4 *r_points,
    __global float8 *r_bounds,
    __global float *dt)
{
    int gid = get_global_id(0);

    // get body from array
    float16 body = bodies[gid];

   	// get acc value and multiply by the timestep do get the displacement vector
   	float2 acc;
   	acc.x = body.s4;
   	acc.y = body.s5;
   	acc.x = acc.x * dt[0];
   	acc.y = acc.y * dt[0];

    // get start/end vertex indices
    int start = (int)body.s7;
    int end   = (int)body.s8;

	// calculate the number of vertices, used later for centroid calculation
	int point_count = end - start + 1;

	// track the center index for the centroid calc at the end as well as the min/max for bounding box
	float x_sum = 0;
	float y_sum = 0;
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

        // subtract prv from pos to get the difference this frame
        float2 diff = pos - prv;
        diff = acc + diff;

        // add friction component
        diff.x *= .980;
        diff.y *= .980;

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

        // update center sum
        x_sum += pos.x;
        y_sum += pos.y;

        // update min/max values for bounding box
        if (pos.x > max_x)
        {
            max_x = pos.x;
        }
        if (pos.x < min_x)
        {
            min_x = pos.x;
        }
        if (pos.y > max_y)
        {
            max_y = pos.y;
        }
        if (pos.y < min_y)
        {
            min_y = pos.y;
        }

        // store updated point in result buffer
        r_points[i] = point;
    }

    // calculate centroid
    body.s0 = x_sum / point_count;
    body.s1 = y_sum / point_count;

    int bound_index = (int)body.s6;

    float8 bounding_box = bounds[bound_index];

    // calculate bounding box
    bounding_box.s0 = min_x;
    bounding_box.s1 = min_y;
    bounding_box.s2 = fabs(max_x - min_x);
    bounding_box.s3 = fabs(max_y - min_y);
    bounding_box.s4 = max_x;
    bounding_box.s5 = max_y;

    r_bounds[bound_index] = bounding_box;

    // store updated body in result buffer
    r_bodies[gid] = body;
}