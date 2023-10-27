

inline void polygon_circle_collision(int polygon_id, int circle_id,
                                     __global float4 *hulls,
                                     __global int2 *hull_flags,
                                     __global int4 *element_tables,
                                     __global float4 *points,
                                     __global float4 *edges,
                                     __global float2 *reactions,
                                     __global int *reaction_index,
                                     __global int *point_reactions,
                                     __global int *counter)
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
    float _distance = polygon_distance(proj_c, proj_p) / (circle.z / 2);
    if (_distance > 0)
    {
        return;
    }
    float abs_distance = fabs(_distance);

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

    bool vs = (vo_f.x & IS_STATIC) !=0;
    bool es = (eo_f.x & IS_STATIC) !=0;
    
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

    float4 vert_point = points[vert_index];
    float4 edge_point_1 = points[edge_index_a];
    float4 edge_point_2 = points[edge_index_b];

    float2 v0 = vert_point.xy;
    float2 e1 = edge_point_1.xy;
    float2 e2 = edge_point_2.xy;

    float2 v0_p = vert_point.zw;
    float2 e1_p = edge_point_1.zw;
    float2 e2_p = edge_point_2.zw;

    float v0_dist = distance(v0, v0_p);
    float e1_dist = distance(e1, e1_p);
    float e2_dist = distance(e2, e2_p);

    // edge reactions
    float contact = edge_contact(e1, e2, v0, collision_vector);

    float edge_scale = 1.0f / (contact * contact + (1 - contact) * (1 - contact));
    float2 e1_reaction = collision_vector * ((1 - contact) * edge_magnitude * edge_scale) * -1.0f;
    float2 e2_reaction = collision_vector * (contact * edge_magnitude * edge_scale) * -1.0f;

    // vertex reaction
    float2 v_reaction = collision_vector * vertex_magnitude;

    if (!vs)
    {
        int i = atomic_inc(&counter[0]);
        reactions[i] = v_reaction;
        reaction_index[i] = vert_index;
        atomic_inc(&point_reactions[vert_index]);
    }
    if (!es)
    {
        int j = atomic_inc(&counter[0]);
        int k = atomic_inc(&counter[0]);
        reactions[j] = e1_reaction;
        reactions[k] = e2_reaction;
        reaction_index[j] = edge_index_a;
        reaction_index[k] = edge_index_b;
        atomic_inc(&point_reactions[edge_index_a]);
        atomic_inc(&point_reactions[edge_index_b]);
    }
}
