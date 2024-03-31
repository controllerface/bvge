

inline void polygon_circle_collision(int polygon_id, 
                                     int circle_id,
                                     __global float4 *hulls,
                                     __global float2 *hull_frictions,
                                     __global int4 *hull_flags,
                                     __global int4 *element_tables,
                                     __global int4 *vertex_tables,
                                     __global float4 *points,
                                     __global int2 *edges,
                                     __global int *edge_flags,
                                     __global float4 *reactions_A,
                                     __global float4 *reactions_B,
                                     __global int *reaction_index,
                                     __global int *reaction_counts,
                                     __global float *masses,
                                     __global int *counter,
                                      float dt)
{
    float4 polygon = hulls[polygon_id];
    float4 circle = hulls[circle_id];
    int4 polygon_table = element_tables[polygon_id];
    int4 circle_table = element_tables[circle_id];

	int polygon_vert_count = polygon_table.y - polygon_table.x + 1;
	int polygon_edge_count = polygon_table.w - polygon_table.z + 1;

    float min_distance = FLT_MAX;

    int vert_hull_id = -1;
    int edge_hull_id = -1;
    int edge_index_a = -1;
    int edge_index_b = -1;
    int vert_index   = -1;
    
    bool invert = false;
    
    float2 collision_normal;
    int4 vertex_table;

    
    // polygon
    for (int point_index = 0; point_index < polygon_edge_count; point_index++)
    {
        int edge_index = polygon_table.z + point_index;
        int2 edge = edges[edge_index];
        int edge_flag = edge_flags[edge_index];
        
        // do not test interior edges
        if (edge_flag == 1) continue;

        int a_index = edge.x;
        int b_index = edge.y;

        float2 va = points[a_index].xy;
        float2 vb = points[b_index].xy;

        float2 normal_buffer = vb - va;

        float xTemp = normal_buffer.y;
        normal_buffer.y = normal_buffer.x * -1;
        normal_buffer.x = xTemp;

        normal_buffer = fast_normalize(normal_buffer);

        float3 proj_a = project_polygon(points, vertex_tables, polygon_table, normal_buffer);
        float3 proj_b = project_circle(circle, normal_buffer);
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
            collision_normal.x = normal_buffer.x;
            collision_normal.y = normal_buffer.y;
            vert_hull_id = circle_id;
            edge_hull_id   = polygon_id;
            min_distance = abs_distance;
            edge_index_a = a_index;
            edge_index_b = b_index;
        }
    }

    // circle check
    int cp_index = closest_point_circle(circle.xy, polygon_table, points, vertex_tables);
    float2 collision_point = points[cp_index].xy;
    float2 edge = collision_point - points[circle_table.x].xy;
    float2 axis = fast_normalize(edge);
    float3 proj_p = project_polygon(points, vertex_tables, polygon_table, axis);
    float3 proj_c = project_circle(circle, axis);
    float _distance = native_divide(polygon_distance(proj_c, proj_p), (native_divide( circle.z, 2)));
    if (_distance > 0)
    {
        return;
    }
    float abs_distance = fabs(_distance);

    if (abs_distance < min_distance)
    {
        collision_normal.x = axis.x;
        collision_normal.y = axis.y;
        min_distance = abs_distance;
    }

    collision_normal = fast_normalize(collision_normal);

    int hull_a_index = circle_id;
    int hull_b_index = polygon_id;

    float4 hull_a = hulls[hull_a_index];
    float4 hull_b = hulls[hull_b_index];

    float2 direction = hull_a.xy - hull_b.xy;
    collision_normal = dot(direction, collision_normal) < 0
        ? collision_normal * -1
        : collision_normal;

    vert_index = circle_table.x;
    min_distance = native_divide(min_distance, fast_length(collision_normal));

    int4 vert_hull_flags = hull_flags[vert_hull_id];
    int4 edge_hull_flags = hull_flags[edge_hull_id];
    float vert_hull_mass = masses[vert_hull_flags.y];
    float edge_hull_mass = masses[edge_hull_flags.y];
    float2 vert_hull_phys = hull_frictions[vert_hull_id];
    float2 edge_hull_phys = hull_frictions[edge_hull_id];

    // collision reaction and opposing direction calculation
    float2 vert_hull_opposing = hull_b.xy - hull_a.xy;
    float2 edge_hull_opposing = hull_a.xy - hull_b.xy;
    
    float total_mass = vert_hull_mass + edge_hull_mass;
    float vert_magnitude = native_divide(edge_hull_mass, total_mass);
    float edge_magnitude = native_divide(vert_hull_mass, total_mass);
    bool static_vert = (vert_hull_flags.x & IS_STATIC) !=0;
    bool static_edge = (edge_hull_flags.x & IS_STATIC) !=0;
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
    float2 collision_vector = collision_normal * min_distance;
    float contact = edge_contact(edge_point_1.xy, edge_point_2.xy, vertex_point.xy, collision_vector);
    float inverse_contact = 1.0f - contact;
    float edge_scale = native_divide(1.0f, (pown(contact, 2) + pown(inverse_contact, 2)));
    float2 edge_1_collision = collision_vector * (inverse_contact * edge_magnitude * edge_scale) * -1;
    float2 edge_2_collision = collision_vector * (contact * edge_magnitude * edge_scale) * -1;
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
            ? vert_hull_phys.x 
            : edge_hull_phys.x
        : max(vert_hull_phys.x, edge_hull_phys.x);

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
            ? vert_hull_phys.y 
            : edge_hull_phys.y
        : max(vert_hull_phys.y, edge_hull_phys.y);

    float2 collision_invert = collision_normal * -1;
    float2 vertex_restitution = restituion_coefficient * dot(vertex_applied_vel, collision_normal) * collision_normal;
    float2 edge_1_restitution = restituion_coefficient * dot(edge_1_applied_vel, collision_invert) * collision_invert;
    float2 edge_2_restitution = restituion_coefficient * dot(edge_2_applied_vel, collision_invert) * collision_invert;

    if (!static_vert)
    {
        int point_index = atomic_inc(&counter[0]);
        float4 vertex_reaction_A = (float4)(vertex_collision, vert_hull_opposing);
        float4 vertex_reaction_B = (float4)(vertex_friction, vertex_restitution);
        reactions_A[point_index] = vertex_reaction_A;
        reactions_B[point_index] = vertex_reaction_B;
        reaction_index[point_index] = vert_index;
        atomic_inc(&reaction_counts[vert_index]);
    }
    if (!static_edge)
    {
        int edge_1_reaction_index = atomic_inc(&counter[0]);
        int edge_2_reaction_index = atomic_inc(&counter[0]);
        float4 edge_1_reaction_A = (float4)(edge_1_collision, edge_hull_opposing);
        float4 edge_1_reaction_B = (float4)(edge_1_friction, edge_1_restitution);
        float4 edge_2_reaction_A = (float4)(edge_2_collision, edge_hull_opposing);
        float4 edge_2_reaction_B = (float4)(edge_2_friction, edge_2_restitution);
        reactions_A[edge_1_reaction_index] = edge_1_reaction_A;
        reactions_A[edge_2_reaction_index] = edge_2_reaction_A;
        reactions_B[edge_1_reaction_index] = edge_1_reaction_B;
        reactions_B[edge_2_reaction_index] = edge_2_reaction_B;
        reaction_index[edge_1_reaction_index] = edge_index_a;
        reaction_index[edge_2_reaction_index] = edge_index_b;
        atomic_inc(&reaction_counts[edge_index_a]);
        atomic_inc(&reaction_counts[edge_index_b]);
    }













    // // vertex and edge object flags
    // int4 vo_f = hull_flags[vert_hull_id];
    // int4 eo_f = hull_flags[edge_hull_id];

    // float2 vo_phys = hull_frictions[vert_hull_id];
    // float2 eo_phys = hull_frictions[edge_hull_id];

    // float2 vo_dir = hull_b.xy - hull_a.xy;
    // float2 eo_dir = hull_a.xy - hull_b.xy;

    // float vo_mass = masses[vo_f.y];
    // float eo_mass = masses[eo_f.y];

    // float total_mass = vo_mass + eo_mass;

    // float2 collision_vector = collision_normal * min_distance;
    
    // float vertex_magnitude = native_divide(eo_mass, total_mass);
    // float edge_magnitude = native_divide(vo_mass, total_mass);

    // bool vs = (vo_f.x & IS_STATIC) !=0;
    // bool es = (eo_f.x & IS_STATIC) !=0;
    
    // bool any_s = (vs || es);

    // vertex_magnitude = any_s 
    //     ? vs ? 0.0f : 1.0f
    //     : vertex_magnitude;

    // edge_magnitude = any_s 
    //     ? es ? 0.0f : 1.0f
    //     : edge_magnitude;

    // float4 vert_point = points[vert_index];
    // float4 edge_point_1 = points[edge_index_a];
    // float4 edge_point_2 = points[edge_index_b];

    // float2 v0 = vert_point.xy;
    // float2 e1 = edge_point_1.xy;
    // float2 e2 = edge_point_2.xy;

    // float2 v0_p = vert_point.zw;
    // float2 e1_p = edge_point_1.zw;
    // float2 e2_p = edge_point_2.zw;

    // float2 v0_dir = v0 - v0_p;
    // float2 e1_dir = e1 - e1_p;
    // float2 e2_dir = e2 - e2_p;

    // float2 v0_v = native_divide(v0_dir, dt);
    // float2 e1_v = native_divide(e1_dir, dt);
    // float2 e2_v = native_divide(e2_dir, dt);

    // float2 v0_rel = v0_v - collision_vector;
    // float2 e1_rel = e1_v - collision_vector;
    // float2 e2_rel = e2_v - collision_vector;

    // float mu = any_s 
    //     ? vs ? vo_phys.x : eo_phys.x
    //     : max(vo_phys.x, eo_phys.x);

    // float2 v0_tan = v0_rel - dot(v0_rel, collision_normal) * collision_normal;
    // float2 e1_tan = e1_rel - dot(e1_rel, collision_normal) * collision_normal;
    // float2 e2_tan = e2_rel - dot(e2_rel, collision_normal) * collision_normal;

    // v0_tan = fast_normalize(v0_tan);
    // e1_tan = fast_normalize(e1_tan);
    // e2_tan = fast_normalize(e2_tan);

    // float2 v0_fric = (-mu * v0_tan) * vertex_magnitude;
    // float2 e1_fric = (-mu * e1_tan) * edge_magnitude;
    // float2 e2_fric = (-mu * e2_tan) * edge_magnitude;

    // // edge reactions_A
    // float contact = edge_contact(e1, e2, v0, collision_vector);
    // float inverse_contact = 1.0f - contact;
    // float edge_scale = native_divide(1.0f, (pown(contact, 2) + pown(inverse_contact, 2)));
    // float2 e1_reaction = collision_vector * ((1 - contact) * edge_magnitude * edge_scale) * -1.0f;
    // float2 e2_reaction = collision_vector * (contact * edge_magnitude * edge_scale) * -1.0f;

    // // vertex reaction
    // float2 v0_reaction = collision_vector * vertex_magnitude;


    // // restitution
    // float2 v0_n = v0 + v0_reaction;
    // float2 e1_n = e1 + e1_reaction;
    // float2 e2_n = e2 + e2_reaction;

    // float2 v0_dir_n = v0_n - v0_p;
    // float2 e1_dir_n = e1_n - e1_p;
    // float2 e2_dir_n = e2_n - e2_p;

    // float2 v0_vn = native_divide(v0_dir_n, dt);
    // float2 e1_vn = native_divide(e1_dir_n, dt);
    // float2 e2_vn = native_divide(e2_dir_n, dt);

    // float ru = any_s 
    //     ? vs ? vo_phys.y : eo_phys.y
    //     : max(vo_phys.y, eo_phys.y);

    // float2 normal_inv = collision_normal * -1;

    // float2 v0_rest = ru * dot(v0_vn, collision_normal) * collision_normal;
    // float2 e1_rest = ru * dot(e1_vn, normal_inv) * normal_inv;
    // float2 e2_rest = ru * dot(e2_vn, normal_inv) * normal_inv;


    // if (!vs)
    // {
    //     int point_index = atomic_inc(&counter[0]);
    //     float4 v0_reaction_4d;
    //     float4 v0_reaction_4d2;
    //     v0_reaction_4d.xy = v0_reaction;
    //     v0_reaction_4d.zw = vo_dir;
    //     v0_reaction_4d2.xy = v0_fric;
    //     v0_reaction_4d2.zw = v0_rest;
    //     reactions_A[point_index] = v0_reaction_4d;
    //     reactions_B[point_index] = v0_reaction_4d2;
    //     reaction_index[point_index] = vert_index;
    //     atomic_inc(&reaction_counts[vert_index]);
    // }
    // if (!es)
    // {
    //     int j = atomic_inc(&counter[0]);
    //     int k = atomic_inc(&counter[0]);
    //     float4 e1_reaction_4d;
    //     float4 e2_reaction_4d;
    //     float4 e1_reaction_4d2;
    //     float4 e2_reaction_4d2;
    //     e1_reaction_4d.xy = e1_reaction;
    //     e1_reaction_4d.zw = eo_dir;
    //     e1_reaction_4d2.xy = e1_fric;
    //     e1_reaction_4d2.zw = e1_rest;
    //     e2_reaction_4d.xy = e2_reaction;
    //     e2_reaction_4d.zw = eo_dir;
    //     e2_reaction_4d2.xy = e2_fric;
    //     e2_reaction_4d2.zw = e2_rest;
    //     reactions_A[j] = e1_reaction_4d;
    //     reactions_A[k] = e2_reaction_4d;
    //     reactions_B[j] = e1_reaction_4d2;
    //     reactions_B[k] = e2_reaction_4d2;
    //     reaction_index[j] = edge_index_a;
    //     reaction_index[k] = edge_index_b;
    //     atomic_inc(&reaction_counts[edge_index_a]);
    //     atomic_inc(&reaction_counts[edge_index_b]);
    // }
}
