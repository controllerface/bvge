// gets the extents of a spatial index for an axis-aligned bounding box
int4 getExtents(int2 corners[])
{
    int4 r;
    r.x = INT_MAX; // min_x
    r.y = INT_MIN; // max_x
    r.z = INT_MAX; // min_y
    r.w = INT_MIN; // max_y
    for (int i = 0; i < sizeof(corners); i++)
    {
        int2 corner = corners[i];
        if (corner.x < r.x)
        {
            r.x = corner.x;
        }
        if (corner.x > r.y)
        {
            r.y = corner.x;
        }
        if (corner.y < r.z)
        {
            r.z = corner.y;
        }
        if (corner.y > r.w)
        {
            r.w = corner.y;
        }
    }
    return r;
}

// calculates a spatial index cell for a given point
int2 getKeyForPoint(float px, float py,
                    float x_spacing, float y_spacing,
                    float x_origin, float y_origin,
                    int x_subdivisions, int y_subdivisions)
{
    float adjusted_x = px - (x_origin);
    float adjusted_y = py - (y_origin);
    int index_x = ((int) floor(adjusted_x / x_spacing));
    int index_y = ((int) floor(adjusted_y / y_spacing));
    int2 out;
    out.x = index_x;
    out.y = index_y;
    return out;
}

__kernel void integrate(
    __global const float16 *bodies,
    __global const float4 *points,
    __global const float8 *bounds,
    __global float16 *r_bodies,
    __global float4 *r_points,
    __global float8 *r_bounds,
    __global float *args)
{
    int gid = get_global_id(0);

    float dt = args[0];
    float x_spacing = args[1];
    float y_spacing = args[2];
    float x_origin = args[3];
    float y_origin = args[4];
    int x_subdivisions = (int) args[5];
    int y_subdivisions = (int) args[6];

    // get body from array
    float16 body = bodies[gid];

   	// get acc value and multiply by the timestep do get the displacement vector
   	float2 acc;
   	acc.x = body.s4;
   	acc.y = body.s5;
   	acc.x = acc.x * dt;
   	acc.y = acc.y * dt;

    // get start/end vertex indices
    int start = (int)body.s7;
    int end   = (int)body.s8;

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

        // subtract prv from pos to get the difference this frame
        float2 diff = pos - prv;
        diff = acc + diff;

        // add friction component todo: take this in as an argument
        diff.x *= .990;
        diff.y *= .990;

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

    // calculate spatial index boundary
    int2 keys[4];

    keys[0] = getKeyForPoint(bounding_box.s0, bounding_box.s1, x_spacing, y_spacing,
        x_origin,
        y_origin,
        x_subdivisions,
        y_subdivisions);

    keys[1] = getKeyForPoint(max_x, bounding_box.s1, x_spacing, y_spacing,
        x_origin,
        y_origin,
        x_subdivisions,
        y_subdivisions);

    keys[2] = getKeyForPoint(max_x, max_y, x_spacing, y_spacing,
        x_origin,
        y_origin,
        x_subdivisions,
        y_subdivisions);

    keys[3] = getKeyForPoint(bounding_box.s0, max_y, x_spacing, y_spacing,
        x_origin,
        y_origin,
        x_subdivisions,
        y_subdivisions);

    int4 k = getExtents(keys);
    body.sb = (float) k.x;
    body.sc = (float) k.y;
    body.sd = (float) k.z;
    body.se = (float) k.w;

    // calculate spatial index key bank size
    int x_count = (k.y - k.x) + 1;
    int y_count = (k.w - k.z) + 1;
    int count = x_count * y_count;
    int size = count * 2;
    body.sf = (float) size;

    // store updated body and bounds data in result buffers
    r_bounds[bound_index] = bounding_box;
    r_bodies[gid] = body;
}