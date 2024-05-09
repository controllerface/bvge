/**
Handles collision between two polygonal hulls
 */
inline void block_collision(int hull_1_id, 
                            int hull_2_id,
                            __global float2 *hulls,
                            __global float *hull_frictions,
                            __global float *hull_restitutions,
                            __global int *hull_armature_ids,
                            __global int *hull_flags,
                            __global int2 *hull_point_tables,
                            __global int2 *hull_edge_tables,
                            __global int *point_flags,
                            __global float4 *points,
                            __global int2 *edges,
                            __global int *edge_flags,
                            __global float8 *reactions,
                            __global int *reaction_index,
                            __global int *reaction_counts,
                            __global float *masses,
                            __global int *counter,
                            float dt)
{
    float2 hull_1 = hulls[hull_1_id];
    float2 hull_2 = hulls[hull_2_id];

    int2 hull_1_point_table = hull_point_tables[hull_1_id];
    int2 hull_1_edge_table = hull_edge_tables[hull_1_id];
    int2 hull_2_point_table = hull_point_tables[hull_2_id];
    int2 hull_2_edge_table = hull_edge_tables[hull_2_id];

	int hull_1_point_count = hull_1_point_table.y - hull_1_point_table.x + 1;
	int hull_1_edge_count  = hull_1_edge_table.y - hull_1_edge_table.x + 1;
	int hull_2_point_count = hull_2_point_table.y - hull_2_point_table.x + 1;
	int hull_2_edge_count  = hull_2_edge_table.y - hull_2_edge_table.x + 1;

    float min_distance = FLT_MAX;

    int vert_hull_id = -1;
    int edge_hull_id = -1;
    int edge_index_a = -1;
    int edge_index_b = -1;
    int vert_index   = -1;
    
    bool invert_hull_order = false;
    
    float2 collision_normal;
    int2 vertex_table;

    // hull 1
    int p1 = 0;

    __attribute__((opencl_unroll_hint(2)))
    for (int point_index = 0; point_index < hull_1_edge_count; point_index++)
    {
        int edge_index = hull_1_edge_table.x + point_index;
        int2 edge = edges[edge_index];
        int edge_flag = edge_flags[edge_index];
        
        // do not test interior edges
        if (edge_flag == 1 || p1 >= 2) continue;
        p1++;

        int a_index = edge.x;
        int b_index = edge.y;

        float2 va = points[a_index].xy;
        float2 vb = points[b_index].xy;

        float2 normal_buffer = vb - va;

        float xTemp = normal_buffer.y;
        normal_buffer.y = normal_buffer.x * -1;
        normal_buffer.x = xTemp;

        normal_buffer = fast_normalize(normal_buffer);

        float3 proj_a = project_polygon(points, point_flags, hull_1_point_table, normal_buffer);
        float3 proj_b = project_polygon(points, point_flags, hull_2_point_table, normal_buffer);
        float distance = polygon_distance(proj_a, proj_b);

        if (distance > 0)
        {
            return;
        }

        float abs_distance = fabs(distance);

        if (abs_distance < min_distance)
        {
            invert_hull_order = true;
            vertex_table = hull_2_point_table;
            collision_normal = normal_buffer;
            vert_hull_id = hull_2_id;
            edge_hull_id = hull_1_id;
            min_distance = abs_distance;
            edge_index_a = a_index;
            edge_index_b = b_index;
        }
    }
    
    // hull 2

    p1 = 0;

    __attribute__((opencl_unroll_hint(2)))
    for (int point_index = 0; point_index < hull_2_edge_count; point_index++)
    {
        int edge_index = hull_2_edge_table.x + point_index;
        int2 edge = edges[edge_index];
        int edge_flag = edge_flags[edge_index];

        // do not test interior edges
        if (edge_flag == 1 || p1 >= 2) continue;
        p1++;

        int a_index = edge.x;
        int b_index = edge.y;

        float2 va = points[a_index].xy;
        float2 vb = points[b_index].xy;
        float2 normal_buffer = vb - va;

        float xTemp = normal_buffer.y;
        normal_buffer.y = normal_buffer.x * -1;
        normal_buffer.x = xTemp;

        normal_buffer = fast_normalize(normal_buffer);

        float3 proj_a = project_polygon(points, point_flags, hull_1_point_table, normal_buffer);
        float3 proj_b = project_polygon(points, point_flags, hull_2_point_table, normal_buffer);
        float distance = polygon_distance(proj_a, proj_b);

        if (distance > 0)
        {
            return;
        }

        float abs_distance = fabs(distance);
        if (abs_distance < min_distance)
        {
            invert_hull_order = false;
            vertex_table = hull_1_point_table;
            collision_normal = normal_buffer;
            vert_hull_id = hull_1_id;
            edge_hull_id = hull_2_id;
            min_distance = abs_distance;
            edge_index_a = a_index;
            edge_index_b = b_index;
        }
    }

    collision_normal = fast_normalize(collision_normal);

    int hull_a_index = invert_hull_order
        ? hull_2_id
        : hull_1_id;

    int hull_b_index = invert_hull_order
        ? hull_1_id
        : hull_2_id;

    float2 hull_a = hulls[hull_a_index];
    float2 hull_b = hulls[hull_b_index];

    float2 direction = hull_a.xy - hull_b.xy;
    collision_normal = dot(direction, collision_normal) < 0
        ? collision_normal * -1
        : collision_normal;

    float3 final_proj = project_polygon(points, point_flags, vertex_table, collision_normal);
    vert_index = final_proj.z;
    min_distance = native_divide(min_distance, fast_length(collision_normal));

    // collision reaction and opposing direction calculation
    float2 vert_hull_opposing = vert_hull_id == hull_1_id 
        ? hull_2.xy - hull_1.xy 
        : hull_1.xy - hull_2.xy;

    float2 edge_hull_opposing = edge_hull_id == hull_1_id 
        ? hull_2.xy - hull_1.xy 
        : hull_1.xy - hull_2.xy;

    int vert_hull_flags = hull_flags[vert_hull_id];
    int edge_hull_flags = hull_flags[edge_hull_id];
    int vert_armature_id = hull_armature_ids[vert_hull_id];
    int edge_armature_id = hull_armature_ids[edge_hull_id];
    float vert_hull_mass = masses[vert_armature_id];
    float edge_hull_mass = masses[edge_armature_id];
    float vert_hull_friction = hull_frictions[vert_hull_id];
    float edge_hull_friction = hull_frictions[edge_hull_id];
    float vert_hull_restitution = hull_restitutions[vert_hull_id];
    float edge_hull_restitution = hull_restitutions[edge_hull_id];
    float total_mass = vert_hull_mass + edge_hull_mass;
    float vert_magnitude = native_divide(edge_hull_mass, total_mass);
    float edge_magnitude = native_divide(vert_hull_mass, total_mass);

    bool static_vert = (vert_hull_flags & IS_STATIC) !=0;
    bool static_edge = (edge_hull_flags & IS_STATIC) !=0;
    bool block_vert = (vert_hull_flags & IS_BLOCK) !=0;
    bool block_edge = (edge_hull_flags & IS_BLOCK) !=0;

    vert_hull_flags = (block_vert && block_edge) 
        ? vert_hull_flags | TOUCH_ALIKE 
        : vert_hull_flags;

    edge_hull_flags = (block_vert && block_edge) 
        ? edge_hull_flags | TOUCH_ALIKE 
        : edge_hull_flags;


    hull_flags[vert_hull_id] = vert_hull_flags;
    hull_flags[edge_hull_id] = edge_hull_flags;

    bool any_static = (static_vert || static_edge);

    vert_magnitude = any_static 
        ? static_vert ? 0.0f : 1.0f
        : vert_magnitude;

    edge_magnitude = any_static 
        ? static_edge ? 0.0f : 1.0f
        : edge_magnitude;

    float4 vertex_point = points[vert_index];
    float4 edge_point_1 = points[edge_index_a];
    float4 edge_point_2 = points[edge_index_b];

    int v_l = vertex_table.x == vert_index 
        ? vertex_table.y 
        : vert_index - 1;

    int v_r = vertex_table.y == vert_index 
        ? vertex_table.x 
        : vert_index + 1;

    float4 vertex_point_l = points[v_l];
    float4 vertex_point_r = points[v_r];
    float4 vertex_point_ex;
    int vertex_ex_index;

    float a_l = angle_between((float4)(vertex_point.xy, vertex_point_l.xy), (float4)(edge_point_1.xy, edge_point_2.xy));
    float a_r = angle_between((float4)(vertex_point.xy, vertex_point_r.xy), (float4)(edge_point_1.xy, edge_point_2.xy));

    bool l_c = fabs(a_l) < .001;
    bool r_c = fabs(a_r) < .001;
    bool v_extend = l_c || r_c;

    vertex_ex_index = v_extend 
        ? l_c
            ? v_l
            : v_r
        : -1;

    vertex_point_ex = v_extend
        ? l_c
            ? vertex_point_l
            : vertex_point_r
        : vertex_point_ex;

    float2 collision_vector = collision_normal * min_distance;

    float2 edge_1_collision;
    float2 edge_2_collision;

    if (v_extend)
    {
        edge_1_collision = collision_vector * -edge_magnitude;
        edge_2_collision = collision_vector * -edge_magnitude;
    }
    else
    {
        float contact = edge_contact(edge_point_1.xy, edge_point_2.xy, vertex_point.xy, collision_vector);
        float inverse_contact = 1.0f - contact;
        float edge_scale = native_divide(1.0f, (pown(contact, 2) + pown(inverse_contact, 2)));
        edge_1_collision = collision_vector * (inverse_contact * edge_magnitude * edge_scale) * -1;
        edge_2_collision = collision_vector * (contact * edge_magnitude * edge_scale) * -1;
    }
    
    float2 vertex_collision = collision_vector * vert_magnitude;

    // friction
    float2 vertex_diff = vertex_point.xy - vertex_point.zw;
    float2 edge_1_diff = edge_point_1.xy - edge_point_1.zw;
    float2 edge_2_diff = edge_point_2.xy - edge_point_2.zw;
    float2 vertex_velocity = native_divide(vertex_diff, dt);
    float2 edge_1_velocity = native_divide(edge_1_diff, dt);
    float2 edge_2_velocity = native_divide(edge_2_diff, dt);
    float2 vertex_rel_vel = vertex_velocity - collision_vector;
    float2 edge_1_rel_vel = edge_1_velocity - collision_vector;
    float2 edge_2_rel_vel = edge_2_velocity - collision_vector;

    float friction_coefficient = any_static 
        ? static_vert 
            ? vert_hull_friction 
            : edge_hull_friction
        : max(vert_hull_friction, edge_hull_friction);

    float2 vertex_tangent = vertex_rel_vel - dot(vertex_rel_vel, collision_normal) * collision_normal;
    float2 edge_1_tangent = edge_1_rel_vel - dot(edge_1_rel_vel, collision_normal) * collision_normal;
    float2 edge_2_tangent = edge_2_rel_vel - dot(edge_2_rel_vel, collision_normal) * collision_normal;
    vertex_tangent = fast_normalize(vertex_tangent);
    edge_1_tangent = fast_normalize(edge_1_tangent);
    edge_2_tangent = fast_normalize(edge_2_tangent);
    float2 vertex_friction = (-friction_coefficient * vertex_tangent) * vert_magnitude;
    float2 edge_1_friction = (-friction_coefficient * edge_1_tangent) * edge_magnitude;
    float2 edge_2_friction = (-friction_coefficient * edge_2_tangent) * edge_magnitude;

    // restitution
    float2 vertex_applied = vertex_point.xy + vertex_collision;
    float2 edge_1_applied = edge_point_1.xy + edge_1_collision;
    float2 edge_2_applied = edge_point_2.xy + edge_2_collision;
    float2 vertex_applied_diff = vertex_applied - vertex_point.zw;
    float2 edge_1_applied_diff = edge_1_applied - edge_point_1.zw;
    float2 edge_2_applied_diff = edge_2_applied - edge_point_2.zw;
    float2 vertex_applied_vel = native_divide(vertex_applied_diff, dt);
    float2 edge_1_applied_vel = native_divide(edge_1_applied_diff, dt);
    float2 edge_2_applied_vel = native_divide(edge_2_applied_diff, dt);

    float restituion_coefficient = any_static 
        ? static_vert 
            ? vert_hull_restitution 
            : edge_hull_restitution
        : max(vert_hull_restitution, edge_hull_restitution);

    float2 collision_invert = collision_normal * -1;
    float2 vertex_restitution = restituion_coefficient * dot(vertex_applied_vel, collision_normal) * collision_normal;
    float2 edge_1_restitution = restituion_coefficient * dot(edge_1_applied_vel, collision_invert) * collision_invert;
    float2 edge_2_restitution = restituion_coefficient * dot(edge_2_applied_vel, collision_invert) * collision_invert;

    if (!static_vert)
    {
        int point_index = atomic_inc(&counter[0]);
        float8 vertex_reactions = (float8)(vertex_collision, vert_hull_opposing, vertex_friction, vertex_restitution);
        reactions[point_index] = vertex_reactions;
        reaction_index[point_index] = vert_index;
        atomic_inc(&reaction_counts[vert_index]);
        if (v_extend)
        {
            int point_index_ex = atomic_inc(&counter[0]);
            float8 vertex_reactions_ex = (float8)(vertex_collision, vert_hull_opposing, vertex_friction, vertex_restitution);
            reactions[point_index_ex] = vertex_reactions_ex;
            reaction_index[point_index_ex] = vertex_ex_index;
            atomic_inc(&reaction_counts[vertex_ex_index]);
        }
    }
    if (!static_edge)
    {
        int edge_1_reaction_index = atomic_inc(&counter[0]);
        int edge_2_reaction_index = atomic_inc(&counter[0]);
        float8 edge_1_reactions = (float8)(edge_1_collision, edge_hull_opposing, edge_1_friction, edge_1_restitution);
        float8 edge_2_reactions = (float8)(edge_2_collision, edge_hull_opposing, edge_2_friction, edge_2_restitution);
        reactions[edge_1_reaction_index] = edge_1_reactions;
        reactions[edge_2_reaction_index] = edge_2_reactions;
        reaction_index[edge_1_reaction_index] = edge_index_a;
        reaction_index[edge_2_reaction_index] = edge_index_b;
        atomic_inc(&reaction_counts[edge_index_a]);
        atomic_inc(&reaction_counts[edge_index_b]);
    }
}
