
inline void polygon_collision(int b1_id, int b2_id,
                             __global float4 *hulls,
                             __global int2 *hull_flags,
                             __global int4 *element_tables,
                             __global float4 *points,
                             __global float4 *edges,
                             __global float2 *reactions,
                             __global int *reaction_index,
                             __global int *point_reactions,
                             __global float *masses,
                             __global int *counter)
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

    float vo_mass = masses[vo_f.y];
    float eo_mass = masses[eo_f.y];

    float total_mass = vo_mass + eo_mass;

    float2 normal = normalBuffer;

    // todo: calculate tangent as well to add friction support
    //  the goal is to find the normalized tangent to the colliding edge, and apply a reaction
    //  that is opposite to the tangent, equal to the component portion of the effective velocity
    //  that is the same direction as the tangent. Note, static objects apply friction to non-statics.
    //  non-statics resolve to largest object, in case of tie, split the difference?
    // the above changes will need to be made to the other collision kernels as well

    float2 collision_vector = normal * min_distance;
    

    // todo: magnitude should be proportional to mass
    // hulls will need to reference the mass of the parent armature to modify the reaction magnitude. Currently, 
    // objects behave as if everything has a mass of 1, splitting reaction magnitudes 50/50. Instead, the mass of 
    // the colliding objects should proportinally scale reactions to be more or less based on the difference between 
    // the masses of the two objects. When two objects with equal mass collide, it should work as it does now, with
    // an even split.
    //
    // the above changes will need to be made to the other collision kernels as well
    
    float vertex_magnitude = eo_mass / total_mass;
    float edge_magnitude = vo_mass / total_mass;

    bool vs = (vo_f.x & IS_STATIC) !=0;
    bool es = (eo_f.x & IS_STATIC) !=0;
    
    // todo: convert to ternary expression for warp efficiency 
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
    float2 e1_reaction = collision_vector * ((1 - contact) * edge_magnitude * edge_scale) * -1;
    float2 e2_reaction = collision_vector * (contact * edge_magnitude * edge_scale) * -1;

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
