inline void circle_collision(int b1_id, int b2_id,
                             __global float4 *hulls,
                             __global int4 *element_tables,
                             __global float4 *points)
{
    float4 hull_1 = hulls[b1_id];
    float4 hull_2 = hulls[b2_id];
    int4 hull_1_table = element_tables[b1_id];
    int4 hull_2_table = element_tables[b2_id];

    float2 normal;
    float depth = 0;
    float _distance = distance(hull_1.xy, hull_2.xy);
    float radii = hull_1.w + hull_2.w;
    if(_distance >= radii)
    {
        return;
    }

    float2 sub = hull_2.xy - hull_1.xy;
    normal = normalize(sub);
    depth = radii - _distance;

    float2 reaction = normal * (float2)(depth + .1);
    float2 offset1 = reaction * (float2)(-0.5);
    float2 offset2 = reaction * (float2)(0.5);

    float4 vert1 = points[hull_1_table.x];
    float4 vert2 = points[hull_2_table.x];
    
    vert1.xy += offset1;
    vert2.xy += offset2;

    points[hull_1_table.x] = vert1;
    points[hull_2_table.x] = vert2;
}


inline void polygon_collision(int b1_id, int b2_id,
                             __global float4 *hulls,
                             __global int2 *hull_flags,
                             __global int4 *element_tables,
                             __global float4 *points,
                             __global float4 *edges)
{

    float4 hull_1 = hulls[b1_id];
    float4 hull_2 = hulls[b2_id];
    int4 hull_1_table = element_tables[b1_id];
    int4 hull_2_table = element_tables[b2_id];

    int start_1 = hull_1_table.x;
    int end_1   = hull_1_table.y;
	int b1_vert_count = end_1 - start_1 + 1;

    int start_2 = hull_2_table.x;
    int end_2   = hull_2_table.y;
	int b2_vert_count = end_2 - start_2 + 1;

    int edge_start_1 = hull_1_table.z;
    int edge_end_1   = hull_1_table.w;
	int b1_edge_count = edge_end_1 - edge_start_1 + 1;

    int edge_start_2 = hull_2_table.z;
    int edge_end_2   = hull_2_table.w;
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

        float3 proj_a = project_polygon(points, hull_1_table, vectorBuffer1);
        float3 proj_b = project_polygon(points, hull_2_table, vectorBuffer1);
        float distance = polygon_distance(proj_a, proj_b);

        if (distance > 0)
        {
            return;
        }

        float abs_distance = fabs(distance);

        if (abs_distance < min_distance)
        {
            invert = true;
            vertex_table = hull_2_table;
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

        float3 proj_a = project_polygon(points, hull_1_table, vectorBuffer1);
        float3 proj_b = project_polygon(points, hull_2_table, vectorBuffer1);
        float distance = polygon_distance(proj_a, proj_b);

        if (distance > 0)
        {
            return;
        }

        float abs_distance = fabs(distance);
        if (abs_distance < min_distance)
        {
            invert = false;
            vertex_table = hull_1_table;
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

    float4 a = hulls[a_idx];
    float4 b = hulls[b_idx];

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
    int2 vo_f = hull_flags[(int)vertex_object_id];
    int2 eo_f = hull_flags[(int)edge_object_id];

    float2 normal = normalBuffer;

    float2 collision_vector = normal * min_distance;
    float vertex_magnitude = .5f;
    float edge_magnitude = .5f;

    bool vs = (vo_f.x & 0x01) !=0;
    bool es = (eo_f.x & 0x01) !=0;
    
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






inline void polygon_circle_collision(int polygon_id, int circle_id,
                                     __global float4 *hulls,
                                     __global int2 *hull_flags,
                                     __global int4 *element_tables,
                                     __global float4 *points,
                                     __global float4 *edges)
{
    float4 polygon = hulls[polygon_id];
    float4 circle = hulls[circle_id];
    int4 polygon_table = element_tables[polygon_id];
    int4 circle_table = element_tables[circle_id];

    int start_1 = polygon_table.x;
    int end_1   = polygon_table.y;
	int b1_vert_count = end_1 - start_1 + 1;

    int edge_start_1 = polygon_table.z;
    int edge_end_1   = polygon_table.w;
	int b1_edge_count = edge_end_1 - edge_start_1 + 1;

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
        float distance = polygon_distance(proj_a, proj_b);

        if (distance > 0)
        {
            return;
        }

        float abs_distance = fabs(distance);

        if (abs_distance < min_distance)
        {
            invert = true;
            vertex_table = circle_table;
            normalBuffer.x = vectorBuffer1.x;
            normalBuffer.y = vectorBuffer1.y;
            vertex_object_id = circle_id;
            edge_object_id   = polygon_id;
            min_distance = abs_distance;
            edge_index_a = a_index;
            edge_index_b = b_index;
        }
    }


    // circle check
    float2 collision_point = points[cp_index].xy;
    float2 edge = collision_point - points[circle_table.x].xy;
    float2 axis = fast_normalize(edge);
    float3 proj_p = project_polygon(points, polygon_table, axis);
    float3 proj_c = project_circle(circle, axis);
    float distance = polygon_distance(proj_c, proj_p) / (circle.z / 2);
    if (distance > 0)
    {
        return;
    }
    float abs_distance = fabs(distance);

    if (abs_distance < min_distance)
    {
        normalBuffer.x = axis.x;
        normalBuffer.y = axis.y;
        min_distance = abs_distance;
    }

    normalBuffer = normalize(normalBuffer);

    int a_idx = circle_id;
    int b_idx = polygon_id;

    float4 a = hulls[a_idx];
    float4 b = hulls[b_idx];

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

    vert_index = circle_table.x;
    min_distance = min_distance / length(normalBuffer);


    // vertex and edge object flags
    int2 vo_f = hull_flags[(int)vertex_object_id];
    int2 eo_f = hull_flags[(int)edge_object_id];

    float2 normal = normalBuffer;

    float2 collision_vector = normal * min_distance;
    float vertex_magnitude = .5f;
    float edge_magnitude = .5f;

    bool vs = (vo_f.x & 0x01) !=0;
    bool es = (eo_f.x & 0x01) !=0;
    
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






/**
Performs collision detection using separating axis theorem, and then applys a reaction
for objects when they are found to be colliding. Reactions detemine one "edge" polygon 
and one "vertex" polygon. The vertex polygon has a single vertex adjusted as a reaction. 
The edge object has two vertices adjusted and the adjustments are in oppostie directions, 
which will naturally apply some degree of rotation to the object.
 todo: add circles, currently assumes polygons 
 */
__kernel void sat_collide(__global int2 *candidates,
                          __global float4 *hulls,
                          __global int4 *element_tables,
                          __global int2 *hull_flags,
                          __global float4 *points,
                          __global float4 *edges)
{
    int gid = get_global_id(0);
    
    int2 current_pair = candidates[gid];
    int b1_id = current_pair.x;
    int b2_id = current_pair.y;
    int2 hull_1_flags = hull_flags[b1_id];
    int2 hull_2_flags = hull_flags[b2_id];
    bool b1s = (hull_1_flags.x & 0x01) !=0;
    bool b2s = (hull_2_flags.x & 0x01) !=0;
    
    if (b1s && b2s) // no collisions between static objects todo: probably can weed these out earlier, during aabb checks
    {
        return;
    }

    bool b1_is_circle = (hull_1_flags.x & 0x02) !=0;
    bool b2_is_circle = (hull_2_flags.x & 0x02) !=0;

    bool b1_is_polygon = (hull_1_flags.x & 0x04) !=0;
    bool b2_is_polygon = (hull_2_flags.x & 0x04) !=0;

    int c_id = b1_is_circle ? b1_id : b2_id;
    int p_id = b1_is_circle ? b2_id : b1_id;

    // todo: it will probably be more performant to have separate kernels for each collision type. There should
    //  be a preliminary kernel that sorts the candidate pairs so they can be run on the right kernel
    if (b1_is_polygon && b2_is_polygon) 
    {
        polygon_collision(b1_id, b2_id, hulls, hull_flags, element_tables, points, edges); 
    }
    else if (b1_is_circle && b2_is_circle) 
    {
        circle_collision(b1_id, b2_id, hulls, element_tables, points); 
    }
    else 
    {
        polygon_circle_collision(p_id, c_id, hulls, hull_flags, element_tables, points, edges); 
    }

    // todo: calculate an epehemeral hull centroid and use it to re-position the armature for both bodies

}
