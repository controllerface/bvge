
inline void polygon_collision(int b1_id, int b2_id,
                             __global float4 *hulls,
                             __global float2 *hull_frictions,
                             __global int4 *hull_flags,
                             __global int4 *element_tables,
                             __global int4 *vertex_tables,
                             __global float4 *points,
                             __global int2 *edges,
                             __global int *edge_flags,
                             __global float4 *reactions,
                             __global float4 *reactions2,
                             __global int *reaction_index,
                             __global int *point_reactions,
                             __global float *masses,
                             __global int *counter,
                             float dt)
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
        int2 edge = edges[edge_index];
        int edge_flag = edge_flags[edge_index];
        
        // do not test interior edges
        if (edge_flag == 1) continue;

        int a_index = edge.x;
        int b_index = edge.y;

        float2 va = points[a_index].xy;
        float2 vb = points[b_index].xy;

        float2 vectorBuffer1 = vb - va;

        float xTemp = vectorBuffer1.y;
        vectorBuffer1.y = vectorBuffer1.x * -1;
        vectorBuffer1.x = xTemp;

        vectorBuffer1 = fast_normalize(vectorBuffer1);

        float3 proj_a = project_polygon(points, vertex_tables, hull_1_table, vectorBuffer1);
        float3 proj_b = project_polygon(points, vertex_tables, hull_2_table, vectorBuffer1);
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
        int2 edge = edges[edge_index];
        int edge_flag = edge_flags[edge_index];

        // do not test interior edges
        if (edge_flag == 1) continue;

        int a_index = edge.x;
        int b_index = edge.y;

        float2 va = points[a_index].xy;
        float2 vb = points[b_index].xy;
        float2 vectorBuffer1 = vb - va;

        float xTemp = vectorBuffer1.y;
        vectorBuffer1.y = vectorBuffer1.x * -1;
        vectorBuffer1.x = xTemp;

        vectorBuffer1 = fast_normalize(vectorBuffer1);

        float3 proj_a = project_polygon(points, vertex_tables, hull_1_table, vectorBuffer1);
        float3 proj_b = project_polygon(points, vertex_tables, hull_2_table, vectorBuffer1);
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

    normalBuffer = fast_normalize(normalBuffer);

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

    float3 final_proj = project_polygon(points, vertex_tables, vertex_table, normalBuffer);
    vert_index = final_proj.z;
    min_distance = native_divide(min_distance, fast_length(normalBuffer));

    // vertex and edge object flags
    int4 vo_f = hull_flags[vertex_object_id];
    int4 eo_f = hull_flags[edge_object_id];

    float2 vo_phys = hull_frictions[vertex_object_id];
    float2 eo_phys = hull_frictions[edge_object_id];

    float2 vo_dir = vertex_object_id == b1_id 
        ? hull_2.xy - hull_1.xy 
        : hull_1.xy - hull_2.xy;

    float2 eo_dir = edge_object_id == b1_id 
        ? hull_2.xy - hull_1.xy 
        : hull_1.xy - hull_2.xy;

    float vo_mass = masses[vo_f.y];
    float eo_mass = masses[eo_f.y];

    float total_mass = vo_mass + eo_mass;

    float2 normal = normalBuffer;
    float2 collision_vector = normal * min_distance;

    float vertex_magnitude = native_divide(eo_mass, total_mass);
    float edge_magnitude = native_divide(vo_mass, total_mass);

    bool vs = (vo_f.x & IS_STATIC) !=0;
    bool es = (eo_f.x & IS_STATIC) !=0;
    
    bool any_s = (vs || es);

    vertex_magnitude = any_s 
        ? vs ? 0.0f : 1.0f
        : vertex_magnitude;

    edge_magnitude = any_s 
        ? es ? 0.0f : 1.0f
        : edge_magnitude;

    float4 vert_point = points[vert_index];
    float4 edge_point_1 = points[edge_index_a];
    float4 edge_point_2 = points[edge_index_b];

    float2 v0 = vert_point.xy;
    float2 e1 = edge_point_1.xy;
    float2 e2 = edge_point_2.xy;

    float2 v0_p = vert_point.zw;
    float2 e1_p = edge_point_1.zw;
    float2 e2_p = edge_point_2.zw;

    float2 v0_dir = v0 - v0_p;
    float2 e1_dir = e1 - e1_p;
    float2 e2_dir = e2 - e2_p;

    float2 v0_v = native_divide(v0_dir, dt);
    float2 e1_v = native_divide(e1_dir, dt);
    float2 e2_v = native_divide(e2_dir, dt);

    float2 v0_rel = v0_v - collision_vector;
    float2 e1_rel = e1_v - collision_vector;
    float2 e2_rel = e2_v - collision_vector;

    float mu = any_s 
        ? vs ? vo_phys.x : eo_phys.x
        : max(vo_phys.x, eo_phys.x);

    float2 v0_tan = v0_rel - dot(v0_rel, normal) * normal;
    float2 e1_tan = e1_rel - dot(e1_rel, normal) * normal;
    float2 e2_tan = e2_rel - dot(e2_rel, normal) * normal;

    v0_tan = fast_normalize(v0_tan);
    e1_tan = fast_normalize(e1_tan);
    e2_tan = fast_normalize(e2_tan);

    float2 v0_fric = (-mu * v0_tan) * vertex_magnitude;
    float2 e1_fric = (-mu * e1_tan) * edge_magnitude;
    float2 e2_fric = (-mu * e2_tan) * edge_magnitude;

    float v_mag_d = vertex_magnitude  * 0.5f;
    float e_mag_d = edge_magnitude  * 0.5f;

    // edge reactions
    float contact = edge_contact(e1, e2, v0, collision_vector);
    float inverse_contact = 1.0f - contact;
    float edge_scale = native_divide(1.0f, (pown(contact, 2) + pown(inverse_contact, 2)));
    float2 e1_reaction = collision_vector * (inverse_contact * edge_magnitude * edge_scale) * -1;
    float2 e2_reaction = collision_vector * (contact * edge_magnitude * edge_scale) * -1;
    float2 e1_reaction_d = collision_vector * (inverse_contact * e_mag_d * edge_scale) * -1;
    float2 e2_reaction_d = collision_vector * (contact * e_mag_d * edge_scale) * -1;



    // vertex reaction
    float2 v0_reaction = collision_vector * vertex_magnitude;
    float2 v0_reaction_d = collision_vector * v_mag_d;


    if (!vs)
    {
        int i = atomic_inc(&counter[0]);
        float4 v0_reaction_4d;
        float4 v0_reaction_4d2;
        v0_reaction_4d.xy = v0_reaction;
        v0_reaction_4d.zw = vo_dir;
        v0_reaction_4d2.xy = v0_fric;
        v0_reaction_4d2.zw = v0_reaction_d;
        reactions[i] = v0_reaction_4d;
        reactions2[i] = v0_reaction_4d2;
        reaction_index[i] = vert_index;
        atomic_inc(&point_reactions[vert_index]);
    }
    if (!es)
    {
        int j = atomic_inc(&counter[0]);
        int k = atomic_inc(&counter[0]);
        float4 e1_reaction_4d;
        float4 e2_reaction_4d;
        float4 e1_reaction_4d2;
        float4 e2_reaction_4d2;
        e1_reaction_4d.xy = e1_reaction;
        e1_reaction_4d.zw = eo_dir;
        e2_reaction_4d.xy = e2_reaction;
        e2_reaction_4d.zw = eo_dir;
        e1_reaction_4d2.xy = e1_fric;
        e2_reaction_4d2.xy = e2_fric;
        e1_reaction_4d2.zw = e1_reaction_d;
        e2_reaction_4d2.zw = e2_reaction_d;
        reactions[j] = e1_reaction_4d;
        reactions[k] = e2_reaction_4d;
        reactions2[j] = e1_reaction_4d2;
        reactions2[k] = e2_reaction_4d2;
        reaction_index[j] = edge_index_a;
        reaction_index[k] = edge_index_b;
        atomic_inc(&point_reactions[edge_index_a]);
        atomic_inc(&point_reactions[edge_index_b]);
    }
}
