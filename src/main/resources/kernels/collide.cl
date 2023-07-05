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

float edgeContact(float2 e1, float2 e2, float2 collision_vertex, float2 collision_vector)
{
    float contact;
    float x_dist = e1.x - e2.x;
    float y_dist = e1.y - e2.y;
    if (fabs(x_dist) > fabs(y_dist))
    {
        float x_offset = (collision_vertex.x - collision_vector.x - e1.x);
        float x_diff = (e2.x - e1.x);
        contact = x_offset / x_diff;
    }
    else
    {
        float y_offset = (collision_vertex.y - collision_vector.y - e1.y);
        float y_diff = (e2.y - e1.y);
        contact = y_offset / y_diff;
    }
    return contact;
}

__kernel void collide(
    __global const int2 *candidates,
    __global const float16 *bodies,
    __global const float4 *points,
    __global float16 *reactions)
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
    manifold.s0 = (float)-1; // vertex object index
    manifold.s1 = (float)-1; // edge obejct index
    manifold.s2 = (float)0;  // normal x
    manifold.s3 = (float)0;  // normal y
    manifold.s4 = FLT_MAX;   // min distance
    manifold.s5 = (float)0;  // edge point A
    manifold.s6 = (float)0;  // edge point B
    manifold.s7 = (float)0;  // vertex point

    // reaction object
    float16 reaction;
    reaction.s0 = -1;

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
            reactions[gid] = reaction;
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
            edge_index_a = a_index;
            edge_index_b = b_index;
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
            reactions[gid] = reaction;
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
            edge_index_a = a_index;
            edge_index_b = b_index;
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

    manifold.s0 = (float)vertex_object_id; // vertex object index
    manifold.s1 = (float)edge_object_id; // edge obejct index
    manifold.s2 = vectorBuffer2.x;  // normal x
    manifold.s3 = vectorBuffer2.y;  // normal y
    manifold.s4 = min_distance;   // min distance
    manifold.s5 = (float)edge_index_a;  // edge point A
    manifold.s6 = (float)edge_index_b;  // edge point B
    manifold.s7 = (float)vert_index;  // vertex point

    reaction.s0 = -1;
    if (manifold.s0 == -1)
    {
        reactions[gid] = reaction;
        return;
    }

    // vertex and edge objects
    float16 vo = bodies[(int)manifold.s0];
    float16 eo = bodies[(int)manifold.s1];
    float2 normal;
    normal.x = manifold.s2;
    normal.y = manifold.s3;

    float2 collision_vector = normal * manifold.s4;
    float vertex_magnitude = .5f;
    float edge_magnitude = .5f;

    // vertex reaction is easy
    float2 v_reaction = collision_vector * vertex_magnitude;

    // now do edge reactions
    float2 e1 = points[(int)manifold.s5].xy;
    float2 e2 = points[(int)manifold.s6].xy;
    float2 collision_vertex = points[(int)manifold.s7].xy;
    float edge_contact = edgeContact(e1, e2, collision_vertex, collision_vector);

    float edge_scale = 1.0f / (edge_contact * edge_contact + (1 - edge_contact) * (1 - edge_contact));
    float2 e1_reaction = collision_vector * ((1 - edge_contact) * edge_magnitude * edge_scale);
    float2 e2_reaction = collision_vector * (edge_contact * edge_magnitude * edge_scale);

    // todo: detemine if everything is needed, if not may fit into smaller data type
    reaction.s0  = (float)manifold.s0;  // vertex object index
    reaction.s1  = (float)manifold.s1;  // edge object index
    reaction.s2  = (float)manifold.s2;  // normal x
    reaction.s3  = (float)manifold.s3;  // normal y
    reaction.s4  = (float)manifold.s4;  // min distance
    reaction.s5  = (float)manifold.s5;  // edge point A
    reaction.s6  = (float)manifold.s6;  // edge point B
    reaction.s7  = (float)manifold.s7;  // vertex point
    reaction.s8  = v_reaction.x;        // vertex object reaction x1
    reaction.s9  = v_reaction.y;        // vertex object reaction y1
    reaction.sa = e1_reaction.x;       // edge object reaction x1
    reaction.sb = e1_reaction.y;       // edge object reaction y1
    reaction.sc = e2_reaction.x;       // edge object reaction x2
    reaction.sd = e2_reaction.y;       // edge object reaction y2
    reaction.se = (float)0;            // [empty]
    reaction.sf = (float)0;            // [empty]

    reactions[gid] = reaction;
}