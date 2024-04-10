/**
Handles collision between one polygonal hull and one circular hull
 */
inline void polygon_circle_collision(int polygon_id, 
                                     int circle_id,
                                     __global float2 *hulls,
                                     __global float2 *hull_scales,
                                     __global float *hull_frictions,
                                     __global float *hull_restitutions,
                                     __global int *hull_armature_ids,
                                     __global int *hull_flags,
                                     __global int4 *element_tables,
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
    float2 polygon = hulls[polygon_id];
    float2 circle = hulls[circle_id];
    float2 circle_scale = hull_scales[circle_id];
    int4 polygon_table = element_tables[polygon_id];
    int4 circle_table = element_tables[circle_id];

	int polygon_vert_count = polygon_table.y - polygon_table.x + 1;
	int polygon_edge_count = polygon_table.w - polygon_table.z + 1;

    float min_distance = FLT_MAX;

    int vert_hull_id = circle_id;
    int edge_hull_id = polygon_id;
    int edge_index_a = -1;
    int edge_index_b = -1;
    int vert_index   = -1;
        
    float2 collision_normal;
    int4 vertex_table = circle_table;

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

        float3 proj_a = project_polygon(points, point_flags, polygon_table, normal_buffer);
        float3 proj_b = project_circle(circle, circle_scale.x, normal_buffer);
        float distance = polygon_distance(proj_a, proj_b);

        if (distance > 0)
        {
            return;
        }

        float abs_distance = fabs(distance);

        if (abs_distance < min_distance)
        {
            collision_normal = normal_buffer;
            min_distance = abs_distance;
            edge_index_a = a_index;
            edge_index_b = b_index;
        }
    }

    // circle check
    int cp_index = closest_point_circle(circle.xy, polygon_table, points, point_flags);
    float2 collision_point = points[cp_index].xy;
    float2 edge = collision_point - points[circle_table.x].xy;
    float2 axis = fast_normalize(edge);
    float3 proj_p = project_polygon(points, point_flags, polygon_table, axis);
    float3 proj_c = project_circle(circle, circle_scale.x, axis);
    float _distance = native_divide(polygon_distance(proj_c, proj_p), (native_divide( circle_scale.x, 2)));
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

    float2 hull_a = hulls[hull_a_index];
    float2 hull_b = hulls[hull_b_index];

    float2 direction = hull_a.xy - hull_b.xy;
    collision_normal = dot(direction, collision_normal) < 0
        ? collision_normal * -1
        : collision_normal;

    vert_index = circle_table.x;
    min_distance = native_divide(min_distance, fast_length(collision_normal));

    // collision reaction and opposing direction calculation
    float2 vert_hull_opposing = hull_b.xy - hull_a.xy;
    float2 edge_hull_opposing = hull_a.xy - hull_b.xy;
    
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
