/**
Performs collision detection using separating axis theorem, and then applys a reaction
for objects when they are found to be colliding. Assumes polygons todo: add circles
Reactions detemine one "edge" polygon and one "vertex" polygon. The vertext polygon
has a single vertex adjusted as a reaction. The edge object has two vertices adjusted
and the adjustments are in oppostie directions, which will naturally apply some
degree of rotation to the object
 */

// todo: convert to float4; transforms

__kernel void sat_collide(__global int2 *candidates,
                          __global float4 *bodies,
                          __global int4 *element_tables,
                          __global int *body_flags,
                          __global float4 *points)
{
    int gid = get_global_id(0);
    
    int2 current_pair = candidates[gid];
    int b1_id = current_pair.x;
    int b2_id = current_pair.y;
    float4 body_1 = bodies[b1_id];
    float4 body_2 = bodies[b2_id];

    int4 body_1_table = element_tables[b1_id];
    int4 body_2_table = element_tables[b2_id];

    int body_1_flags = body_flags[b1_id];
    int body_2_flags = body_flags[b2_id];

    bool b1s = (body_1_flags && 0x01) !=0;
    bool b2s = (body_2_flags && 0x01) !=0;
    
    if (b1s && b2s) // these should be filtered before getting here, but just in case..
    {
        return;
    }

    int start_1 = body_1_table.x;
    int end_1   = body_1_table.y;
	int b1_vert_count = end_1 - start_1 + 1;

    int start_2 = body_2_table.x;
    int end_2   = body_2_table.y;
	int b2_vert_count = end_2 - start_2 + 1;

    float min_distance   = FLT_MAX;
    int vertex_object_id = -1;
    int edge_object_id   = -1;
    int edge_index_a     = -1;
    int edge_index_b     = -1;
    int vert_index       = -1;
    bool invert          = false;
    
    float2 normalBuffer;
    int4 vertex_table;

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

        float3 proj_a = project_polygon(points, body_1_table, vectorBuffer1);
        float3 proj_b = project_polygon(points, body_2_table, vectorBuffer1);
        float distance = polygon_distance(proj_a, proj_b);

        if (distance > 0)
        {
            return;
        }

        float abs_distance = fabs(distance);

        if (abs_distance < min_distance)
        {
            invert = true;
            vertex_table = body_2_table;
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

        float3 proj_a = project_polygon(points, body_1_table, vectorBuffer1);
        float3 proj_b = project_polygon(points, body_2_table, vectorBuffer1);
        float distance = polygon_distance(proj_a, proj_b);

        if (distance > 0)
        {
            return;
        }

        float abs_distance = fabs(distance);
        if (abs_distance < min_distance)
        {
            invert = false;
            vertex_table = body_1_table;
            normalBuffer.x = vectorBuffer1.x;
            normalBuffer.y = vectorBuffer1.y;
            vertex_object_id = (float)b1_id;
            edge_object_id   = (float)b2_id;
            min_distance = abs_distance;
            edge_index_a = a_index;
            edge_index_b = b_index;
        }
    }

    float3 pr = project_polygon(points, vertex_table, normalBuffer);
    vert_index = pr.z;
    min_distance = min_distance / length(normalBuffer);
    normalBuffer = normalize(normalBuffer);

    float4 a = (invert)
        ? body_2
        : body_1;

    float4 b = (invert)
        ? body_1
        : body_2;

    float2 transformA;
    transformA.x = a.x;
    transformA.y = a.y;

    float2 transformB;
    transformB.x = b.x;
    transformB.y = b.y;

    float2 direction = transformA - transformB;

    float dirdot = (float)dot(direction, normalBuffer);
    if (dirdot < 0)
    {
        normalBuffer.x = normalBuffer.x * -1;
        normalBuffer.y = normalBuffer.y * -1;
    }

    if (vertex_object_id == -1)
    {
        return;
    }

    // vertex and edge object flags
    int vo_f = body_flags[(int)vertex_object_id];
    int eo_f = body_flags[(int)edge_object_id];

    float2 normal = normalBuffer;

    float2 collision_vector = normal * min_distance;
    float vertex_magnitude = .5f;
    float edge_magnitude = .5f;

    bool vs = (vo_f && 0x01) !=0;
    bool es = (eo_f && 0x01) !=0;
    
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

    // todo: this should techncially be atomic, however visually it doesn't
    //  seem to matter right now. probably should do it "right" though at some point
    points[vert_index].x += v_reaction.x;
    points[vert_index].y += v_reaction.y;
    points[edge_index_a].x -= e1_reaction.x;
    points[edge_index_a].y -= e1_reaction.y;
    points[edge_index_b].x -= e2_reaction.x;
    points[edge_index_b].y -= e2_reaction.y;

    // uncomment below for "fake" inelastic collisions
    // todo: previous location should be updated so relative difference is the same as before,
    //  but reaction difference is "cancelled out"
    // points[vert_index].z = points[vert_index].x - FLT_EPSILON;
    // points[vert_index].w = points[vert_index].y - FLT_EPSILON;
    // points[edge_index_a].z = points[edge_index_a].x + FLT_EPSILON;
    // points[edge_index_a].w = points[edge_index_a].y + FLT_EPSILON;
    // points[edge_index_b].z = points[edge_index_b].x + FLT_EPSILON;
    // points[edge_index_b].w = points[edge_index_b].y + FLT_EPSILON;
}
