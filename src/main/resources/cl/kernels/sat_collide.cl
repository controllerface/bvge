inline void polygon_collision(int b1_id, int b2_id,
                             __global float4 *bodies,
                             __global int *body_flags,
                             __global int4 *element_tables,
                             __global float4 *points,
                             __global float4 *edges)
{

    float4 body_1 = bodies[b1_id];
    float4 body_2 = bodies[b2_id];
    int4 body_1_table = element_tables[b1_id];
    int4 body_2_table = element_tables[b2_id];

    int start_1 = body_1_table.x;
    int end_1   = body_1_table.y;
	int b1_vert_count = end_1 - start_1 + 1;

    int start_2 = body_2_table.x;
    int end_2   = body_2_table.y;
	int b2_vert_count = end_2 - start_2 + 1;

    int edge_start_1 = body_1_table.z;
    int edge_end_1   = body_1_table.w;
	int b1_edge_count = edge_end_1 - edge_start_1 + 1;

    int edge_start_2 = body_2_table.z;
    int edge_end_2   = body_2_table.w;
	int b2_edge_count = edge_end_2 - edge_start_2 + 1;

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
    for (int i = 0; i < b1_edge_count; i++)
    {
        int edge_index = edge_start_1 + i;
        float4 edge = edges[edge_index];
        
        // do not test interior edges
        if (edge.w == 1) continue;

        int a_index = edge.x;
        int b_index = edge.y;

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
            vertex_object_id = b2_id;
            edge_object_id   = b1_id;
            min_distance = abs_distance;
            edge_index_a = a_index;
            edge_index_b = b_index;
        }
    }
    
    // object 2
    for (int i = 0; i < b2_edge_count; i++)
    {
        int edge_index = edge_start_2 + i;
        float4 edge = edges[edge_index];
        
        // do not test interior edges
        if (edge.w == 1) continue;

        int a_index = edge.x;
        int b_index = edge.y;

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
            vertex_object_id = b1_id;
            edge_object_id   = b2_id;
            min_distance = abs_distance;
            edge_index_a = a_index;
            edge_index_b = b_index;
        }
    }

    normalBuffer = normalize(normalBuffer);

    int a_idx = (invert)
        ? b2_id
        : b1_id;

    int b_idx = (invert)
        ? b1_id
        : b2_id;

    float4 a = bodies[a_idx];
    float4 b = bodies[b_idx];

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

    float3 final_proj = project_polygon(points, vertex_table, normalBuffer);
    vert_index = final_proj.z;
    min_distance = min_distance / length(normalBuffer);


    // vertex and edge object flags
    int vo_f = body_flags[(int)vertex_object_id];
    int eo_f = body_flags[(int)edge_object_id];

    float2 normal = normalBuffer;

    float2 collision_vector = normal * min_distance;
    float vertex_magnitude = .5f;
    float edge_magnitude = .5f;

    bool vs = (vo_f & 0x01) !=0;
    bool es = (eo_f & 0x01) !=0;
    
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

    float2 e1 = points[edge_index_a].xy;
    float2 e2 = points[edge_index_b].xy;
    float2 collision_vertex = points[vert_index].xy;

    // edge reactions
    float contact = edge_contact(e1, e2, collision_vertex, collision_vector);

    float edge_scale = 1.0f / (contact * contact + (1 - contact) * (1 - contact));
    float2 e1_reaction = collision_vector * ((1 - contact) * edge_magnitude * edge_scale);
    float2 e2_reaction = collision_vector * (contact * edge_magnitude * edge_scale);

    // vertex reaction
    float2 v_reaction = collision_vector * vertex_magnitude;

    // todo: this should technically be atomic, however visually it doesn't
    //  seem to matter right now. probably should do it "right" though at some point
    points[vert_index].x += v_reaction.x;
    points[vert_index].y += v_reaction.y;
    points[edge_index_a].x -= e1_reaction.x;
    points[edge_index_a].y -= e1_reaction.y;
    points[edge_index_b].x -= e2_reaction.x;
    points[edge_index_b].y -= e2_reaction.y;
}

inline void circle_collision(int b1_id, int b2_id,
                             __global float4 *bodies,
                             __global int4 *element_tables,
                             __global float4 *points)
{
    float4 body_1 = bodies[b1_id];
    float4 body_2 = bodies[b2_id];
    int4 body_1_table = element_tables[b1_id];
    int4 body_2_table = element_tables[b2_id];

    float2 normal;
    float depth = 0;
    float _distance = distance(body_1.xy, body_2.xy);
    float radii = body_1.w + body_2.w;
    if(_distance >= radii)
    {
        return;
    }

    float2 sub = body_2.xy - body_1.xy;
    normal = normalize(sub);
    depth = radii - _distance;

    float2 reaction = normal * (float2)(depth + .1);
    float2 offset1 = reaction * (float2)(-0.5);
    float2 offset2 = reaction * (float2)(0.5);

    float4 vert1 = points[body_1_table.x];
    float4 vert2 = points[body_2_table.x];
    
    vert1.xy += offset1;
    vert2.xy += offset2;

    points[body_1_table.x] = vert1;
    points[body_2_table.x] = vert2;
}





inline void polygon_circle_collision(int b1_id, int b2_id,
                                     __global float4 *bodies,
                                     __global int *body_flags,
                                     __global int4 *element_tables,
                                     __global float4 *points,
                                     __global float4 *edges)
{

    float4 polygon = bodies[b1_id];
    float4 circle = bodies[b2_id];
    int4 polygon_table = element_tables[b1_id];
    int4 circle_table = element_tables[b2_id];

    int start_1 = polygon_table.x;
    int end_1   = polygon_table.y;
	int b1_vert_count = end_1 - start_1 + 1;

    int start_2 = circle_table.x;
    int end_2   = circle_table.y;
	int b2_vert_count = end_2 - start_2 + 1;

    int edge_start_1 = polygon_table.z;
    int edge_end_1   = polygon_table.w;
	int b1_edge_count = edge_end_1 - edge_start_1 + 1;

    int edge_start_2 = circle_table.z;
    int edge_end_2   = circle_table.w;
	int b2_edge_count = edge_end_2 - edge_start_2 + 1;

    float min_distance   = FLT_MAX;
    int vertex_object_id = -1;
    int edge_object_id   = -1;
    int edge_index_a     = -1;
    int edge_index_b     = -1;
    int vert_index       = -1;
    bool invert          = false;
    
    float2 normalBuffer;
    int4 vertex_table;

    int cp_index = closest_point_circle(circle.xy, polygon_table, points);
    float4 circle_point = points[start_2];

    // polygon
    for (int i = 0; i < b1_edge_count; i++)
    {
        int edge_index = edge_start_1 + i;
        float4 edge = edges[edge_index];
        
        // do not test interior edges
        if (edge.w == 1) continue;

        int a_index = edge.x;
        int b_index = edge.y;

        float2 va = points[a_index].xy;
        float2 vb = points[b_index].xy;

        float2 vectorBuffer1 = vb - va;

        float xTemp = vectorBuffer1.y;
        vectorBuffer1.y = vectorBuffer1.x * -1;
        vectorBuffer1.x = xTemp;

        vectorBuffer1 = fast_normalize(vectorBuffer1);

        float3 proj_a = project_polygon(points, polygon_table, vectorBuffer1);
        float3 proj_b = project_circle(circle, vectorBuffer1);
        float _distance = polygon_distance(proj_a, proj_b);

        if (_distance > 0)
        {
            //printf("bail polgon");
            return;
        }

        float abs_distance = fabs(_distance);

        if (abs_distance < min_distance)
        {
            invert = true;
            vertex_table = circle_table;
            normalBuffer.x = vectorBuffer1.x;
            normalBuffer.y = vectorBuffer1.y;
            vertex_object_id = b2_id;
            edge_object_id   = b1_id;
            min_distance = abs_distance;
            edge_index_a = a_index;
            edge_index_b = b_index;
        }
    }
    
    // circle
    // todo: try this part again. this part of
    //  of the check in the original JS impl
    
    // confirmed the collision point index is correctly determined 
    //printf("debug CP index %d", cp_index);
    float2 collision_point = points[cp_index].xy;
    float2 edge = collision_point - circle_point.xy;
    float2 axis;
    axis.x = edge.y;
    axis.y = edge.x * -1;
    axis = fast_normalize(axis);

    float3 polygon_projection = project_polygon(points, polygon_table, axis);
    float3 circle_projection = project_circle(circle, axis);

    if (polygon_projection.x >= circle_projection.y || circle_projection.x >= polygon_projection.y)
    {
        return;
    }

    float depth = max(circle_projection.y - polygon_projection.x, polygon_projection.y - circle_projection.x) / circle.w;
    printf("debug depth: %f", depth);
    float abs_distance = fabs(depth);
    if (abs_distance < min_distance)
    {
        invert = false;
        min_distance = abs_distance;
        normalBuffer.x = axis.x;
        normalBuffer.y = axis.y;
    }

    normalBuffer = normalize(normalBuffer);

    int a_idx = (invert)
        ? b2_id
        : b1_id;

    int b_idx = (invert)
        ? b1_id
        : b2_id;

    float4 a = bodies[a_idx];
    float4 b = bodies[b_idx];

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

    //float3 final_proj = project_polygon(points, vertex_table, normalBuffer);
    vert_index = start_2;
    min_distance = min_distance / length(normalBuffer);


    // vertex and edge object flags
    //int vo_f = body_flags[(int)vertex_object_id];
    //int eo_f = body_flags[(int)edge_object_id];

    float2 normal = normalBuffer;

    float2 collision_vector = normal * min_distance;
    float vertex_magnitude = 1.0f;
    // float edge_magnitude = .5f;

    // bool vs = (vo_f & 0x01) !=0;
    // bool es = (eo_f & 0x01) !=0;
    
    // if (vs || es)
    // {
    //     if (vs)
    //     {
    //         vertex_magnitude = 0.0f;
    //         edge_magnitude = 1.0f;
    //     }
    //     if (es)
    //     {
    //         vertex_magnitude = 1.1f;
    //         edge_magnitude = 0.0f;
    //     }
    // }

    //if (invert)
    //{
        //float2 e1 = points[edge_index_a].xy;
        //float2 e2 = points[edge_index_b].xy;
        //float2 collision_vertex = points[vert_index].xy;

        // edge reactions
        //float contact = edge_contact(e1, e2, collision_vertex, collision_vector);

        //float edge_scale = 1.0f / (contact * contact + (1 - contact) * (1 - contact));
        //float2 e1_reaction = collision_vector * ((1 - contact) * edge_magnitude * edge_scale);
        //float2 e2_reaction = collision_vector * (contact * edge_magnitude * edge_scale);

        // vertex reaction
        float2 v_reaction = collision_vector * vertex_magnitude;
        //float2 e_reaction = collision_vector * edge_magnitude;

        // todo: this should technically be atomic, however visually it doesn't
        //  seem to matter right now. probably should do it "right" though at some point
        points[vert_index].x += v_reaction.x;
        points[vert_index].y += v_reaction.y;
        //points[edge_index_a].x -= e1_reaction.x;
        //points[edge_index_a].y -= e1_reaction.y;
        //points[edge_index_b].x -= e2_reaction.x;
        //points[edge_index_b].y -= e2_reaction.y;
    //}
    // else
    // {
    //     float2 v_reaction = collision_vector * vertex_magnitude;
    //     float2 e_reaction = collision_vector * edge_magnitude;
    //     points[cp_index].x += v_reaction.x;
    //     points[cp_index].y += v_reaction.y;
    //     points[start_2].x -= e_reaction.x;
    //     points[start_2].y -= e_reaction.y;
    // }
}






/**
Performs collision detection using separating axis theorem, and then applys a reaction
for objects when they are found to be colliding. Reactions detemine one "edge" polygon 
and one "vertex" polygon. The vertex polygon has a single vertex adjusted as a reaction. 
The edge object has two vertices adjusted and the adjustments are in oppostie directions, 
which will naturally apply some degree of rotation to the object.
 todo: add circles, currently assumes polygons 
 */
__kernel void sat_collide(__global int2 *candidates,
                          __global float4 *bodies,
                          __global int4 *element_tables,
                          __global int *body_flags,
                          __global float4 *points,
                          __global float4 *edges)
{
    int gid = get_global_id(0);
    
    int2 current_pair = candidates[gid];
    int b1_id = current_pair.x;
    int b2_id = current_pair.y;
    int body_1_flags = body_flags[b1_id];
    int body_2_flags = body_flags[b2_id];
    bool b1s = (body_1_flags & 0x01) !=0;
    bool b2s = (body_2_flags & 0x01) !=0;
    
    if (b1s && b2s) // no collisions between static objects todo: probably can weed these out earlier, during aabb checks
    {
        return;
    }

    bool b1_is_circle = (body_1_flags & 0x02) !=0;
    bool b2_is_circle = (body_2_flags & 0x02) !=0;

    bool b1_is_polygon = (body_1_flags & 0x04) !=0;
    bool b2_is_polygon = (body_2_flags & 0x04) !=0;

    // no circles for now
    // if (b1_is_circle || b2_is_circle)
    // {
    //     return;
    // }

    // todo: it will probably be more performant to have separate kernels for each collision type. There should
    //  be a prelimianry kernel that sorts the candidate pairs so they can be 
    if (b1_is_polygon && b2_is_polygon) polygon_collision(b1_id, b2_id, bodies, body_flags, element_tables, points, edges); 
    else if (b1_is_circle && b2_is_circle) circle_collision(b1_id, b2_id, bodies, element_tables, points); 
    else 
    {
        int c_id = b1_is_circle ? b1_id : b2_id;
        int p_id = b1_is_circle ? b2_id : b1_id;
        polygon_circle_collision(p_id, c_id, bodies, body_flags, element_tables, points, edges); 
    }
}
