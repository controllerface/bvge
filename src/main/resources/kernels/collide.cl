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

    bool b1s = (body_1.s6 && 0x01) !=0;
    bool b2s = (body_2.s6 && 0x01) !=0;
    
    if (b1s && b2s) // these should be filtered before getting here, but just in case..
    {
        return;
    }

    int start_1 = (int)body_1.s7;
    int end_1   = (int)body_1.s8;
	int b1_vert_count = end_1 - start_1 + 1;

    int start_2 = (int)body_2.s7;
    int end_2   = (int)body_2.s8;
	int b2_vert_count = end_2 - start_2 + 1;

    float min_distance   = FLT_MAX;
    int vertex_object_id = -1;
    int edge_object_id   = -1;
    int edge_index_a     = -1;
    int edge_index_b     = -1;
    int vert_index       = -1;
    bool invert          = false;
    
    float2 normalBuffer;
    float16 vertex_body;
 
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

        float3 proj_a = project_polygon(points, body_1, vectorBuffer1);
        float3 proj_b = project_polygon(points, body_2, vectorBuffer1);
        float distance = polygon_distance(proj_a, proj_b);

        if (distance > 0)
        {
            reactions[gid] = reaction;
            return;
        }

        float abs_distance = fabs(distance);

        if (abs_distance < min_distance)
        {
            invert = true;
            vertex_body = body_2;
            normalBuffer.x = vectorBuffer1.x;
            normalBuffer.y = vectorBuffer1.y;
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

        float3 proj_a = project_polygon(points, body_1, vectorBuffer1);
        float3 proj_b = project_polygon(points, body_2, vectorBuffer1);
        float distance = polygon_distance(proj_a, proj_b);

        if (distance > 0)
        {
            reactions[gid] = reaction;
            return;
        }

        float abs_distance = fabs(distance);
        if (abs_distance < min_distance)
        {
            invert = false;
            vertex_body = body_1;
            normalBuffer.x = vectorBuffer1.x;
            normalBuffer.y = vectorBuffer1.y;
            vertex_object_id = (float)b1_id;
            edge_object_id   = (float)b2_id;
            min_distance = abs_distance;
            edge_index_a = a_index;
            edge_index_b = b_index;
        }
    }

    float3 pr = project_polygon(points, vertex_body, normalBuffer);
    vert_index = pr.z;
    min_distance = min_distance / length(normalBuffer);
    normalBuffer = normalize(normalBuffer);

    float16 a = (invert)
        ? body_2
        : body_1;

    float16 b = (invert)
        ? body_1
        : body_2;

    float2 transformA;
    transformA.x = a.s0;
    transformA.y = a.s1;

    float2 transformB;
    transformB.x = b.s0;
    transformB.y = b.s1;

    float2 direction = transformA - transformB;

    float dirdot = (float)dot(direction, normalBuffer);
    if (dirdot < 0)
    {
        normalBuffer.x = normalBuffer.x * -1;
        normalBuffer.y = normalBuffer.y * -1;
    }

    reaction.s0 = -1;
    if (vertex_object_id == -1)
    {
        reactions[gid] = reaction;
        return;
    }

    // vertex and edge objects
    float16 vo = bodies[(int)vertex_object_id];
    float16 eo = bodies[(int)edge_object_id];
    float2 normal = normalBuffer;

    float2 collision_vector = normal * min_distance;
    // todo: check body flags for static geometry, always default to 0 impact for the static
    //  body, and 1.0 for the non-static body.
    float vertex_magnitude = .5f;
    float edge_magnitude = .5f;

    bool vs = (vo.s6 && 0x01) !=0;
    bool es = (eo.s6 && 0x01) !=0;
    
    if (vs || es)
    {
        if (vs)
        {
            vertex_magnitude = 0.0f;
            edge_magnitude = 1.0f;
        }
        if (es)
        {
            vertex_magnitude = 1.0f;
            edge_magnitude = 0.0f;
        }
    }

    // vertex reaction is easy
    float2 v_reaction = collision_vector * vertex_magnitude;

    // now do edge reactions
    float2 e1 = points[edge_index_a].xy;
    float2 e2 = points[edge_index_b].xy;
    float2 collision_vertex = points[vert_index].xy;
    float contact = edge_contact(e1, e2, collision_vertex, collision_vector);

    float edge_scale = 1.0f / (contact * contact + (1 - contact) * (1 - contact));
    float2 e1_reaction = collision_vector * ((1 - contact) * edge_magnitude * edge_scale);
    float2 e2_reaction = collision_vector * (contact * edge_magnitude * edge_scale);

    reaction.s0 = (float)vert_index;
    reaction.s1 = (float)edge_index_a;
    reaction.s2 = (float)edge_index_b;
    reaction.s3 = v_reaction.x;  
    reaction.s4 = v_reaction.y;  
    reaction.s5 = e1_reaction.x;  
    reaction.s6 = e1_reaction.y;  
    reaction.s7 = e2_reaction.x;  
    reaction.s8 = e2_reaction.y;   
    // reaction.s9 through reaction.sf are currently empty, but may be used later 
    // as collision checks are updated with more complex logic

    reactions[gid] = reaction;
}
