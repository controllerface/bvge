float3 projectPolygon(__global const float4 *points, float16 body, float2 normal)
{
    int start = (int)body.s7;
    int end   = (int)body.s8;
	int vert_count = end - start + 1;

    float3 result;
    result.x = (float)0; // min
    result.y = (float)0; // max
    result.z = (float)0; // index
    bool minYet = false;
    bool maxYet = false;
    for (int i = 0; i < vert_count; i++)
    {
        int n = start + i;
        float2 v = points[n].xy;
        float proj = dot(v, normal);
        if (proj < result.x || !minYet)
        {
            result.x = proj;
            result.z = n;
            minYet = true;
        }
        if (proj > result.y || !maxYet)
        {
            result.y = proj;
            maxYet = true;
        }
    }
    return result;
}

float polygonDistance(float3 proj_a, float3 proj_b)
{
    if (proj_a.x < proj_b.x)
    {
        return proj_b.x - proj_a.y;
    }
    else
    {
        return proj_a.x - proj_b.y;
    }
}

__kernel void collide(
    __global const int2 *candidates,
    __global const float16 *bodies,
    __global const float4 *points,
    __global float8 *manifolds)
{
    int gid = get_global_id(0);
    
    int2 current_pair = candidates[gid];
    int b1_id = current_pair.x;
    int b2_id = current_pair.y;
    float16 body_1 = bodies[b1_id];
    float16 body_2 = bodies[b2_id];

    int start_1 = (int)body_1.s7;
    int end_1   = (int)body_1.s8;
	int b1_vert_count = end_1 - start_1 + 1;

    int start_2 = (int)body_2.s7;
    int end_2   = (int)body_2.s8;
	int b2_vert_count = end_2 - start_2 + 1;

    float min_distance = FLT_MAX;
    int vertex_object_id = -1;
    int edge_object_id   = -1;
    int edge_index_a     = -1;
    int edge_index_b     = -1;
    int vert_index       = -1;
    int invert = 0;
    float2 vectorBuffer2;
    float16 vertex_body;
 
    // manifold object
    float8 manifold;
    manifold[0] = (float)-1; // vertex object index
    manifold[1] = (float)-1; // edge obejct index
    manifold[2] = (float)0;  // normal x
    manifold[3] = (float)0;  // normal y
    manifold[4] = FLT_MAX;   // min distance
    manifold[5] = (float)0;  // edge point A
    manifold[6] = (float)0;  // edge point B
    manifold[7] = (float)0;  // vertex point

    // object 1
    for (int i = 0; i < b1_vert_count; i++)
    {
        int a_index = start_1 + i;
        int b_index = a_index < end_1 
            ? a_index + 1
            : start_1;

        float2 va = points[a_index].xy;
        float2 vb = points[b_index].xy;

        float2 vectorBuffer1 = vb - va;

        float xTemp = vectorBuffer1.y;
        vectorBuffer1.y = vectorBuffer1.x * -1;
        vectorBuffer1.x = xTemp;

        vectorBuffer1 = fast_normalize(vectorBuffer1);

        float3 proj_a = projectPolygon(points, body_1, vectorBuffer1);
        float3 proj_b = projectPolygon(points, body_2, vectorBuffer1);
        float distance = polygonDistance(proj_a, proj_b);

        if (distance > 0)
        {
            manifolds[gid] = manifold;
            return;
        }

        float abs_distance = fabs(distance);

        if (abs_distance < min_distance)
        {
            invert = 1;
            vertex_body = body_2;
            vectorBuffer2.x = vectorBuffer1.x;
            vectorBuffer2.y = vectorBuffer1.y;
            vertex_object_id = (float)b2_id;
            edge_object_id   = (float)b1_id;
            min_distance = abs_distance;
            edge_index_a = i;
            edge_index_b = b_index == start_1 ? 0 : i + 1;
        }
    }

    // object 2
    for (int i = 0; i < b2_vert_count; i++)
    {
        int a_index = start_2 + i;
        int b_index = a_index < end_2 
            ? a_index + 1
            : start_2;

        float2 va = points[a_index].xy;
        float2 vb = points[b_index].xy;
        float2 vectorBuffer1 = vb - va;

        float xTemp = vectorBuffer1.y;
        vectorBuffer1.y = vectorBuffer1.x * -1;
        vectorBuffer1.x = xTemp;

        vectorBuffer1 = fast_normalize(vectorBuffer1);

        float3 proj_a = projectPolygon(points, body_1, vectorBuffer1);
        float3 proj_b = projectPolygon(points, body_2, vectorBuffer1);
        float distance = polygonDistance(proj_a, proj_b);

        if (distance > 0)
        {
            manifolds[gid] = manifold;
            return;
        }

        float abs_distance = fabs(distance);
        if (abs_distance < min_distance)
        {
            invert = 0;
            vertex_body = body_1;
            vectorBuffer2.x = vectorBuffer1.x;
            vectorBuffer2.y = vectorBuffer1.y;
            vertex_object_id = (float)b1_id;
            edge_object_id   = (float)b2_id;
            min_distance = abs_distance;
            edge_index_a = i;
            edge_index_b = b_index == start_2 ? 0 : i + 1;
        }
    }

    float3 pr = projectPolygon(points, vertex_body, vectorBuffer2);
    vert_index = pr.z;
    min_distance = min_distance / length(vectorBuffer2);
    vectorBuffer2 = normalize(vectorBuffer2);

    float16 a = (invert == 1)
        ? body_2
        : body_1;

    float16 b = (invert ==  1)
        ? body_1
        : body_2;

    float2 transformA;
    transformA.x = a.s0;
    transformA.y = a.s1;

    float2 transformB;
    transformB.x = b.s0;
    transformB.y = b.s1;

    float2 direction = transformA - transformB;

    float dirdot = (float)dot(direction, vectorBuffer2);
    if (dirdot < 0)
    {
        vectorBuffer2.x = vectorBuffer2.x * -1;
        vectorBuffer2.y = vectorBuffer2.y * -1;
    }

    manifold[0] = (float)vertex_object_id; // vertex object index
    manifold[1] = (float)edge_object_id; // edge obejct index
    manifold[2] = vectorBuffer2.x;  // normal x
    manifold[3] = vectorBuffer2.y;  // normal y
    manifold[4] = min_distance;   // min distance
    manifold[5] = (float)edge_index_a;  // edge point A
    manifold[6] = (float)edge_index_b;  // edge point B
    manifold[7] = (float)vert_index;  // vertex point

    manifolds[gid] = manifold;
}