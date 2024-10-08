__kernel void count_mesh_instances(__global int *hull_mesh_ids,
                                   __global int *hull_flags,
                                   __global int *hull_entity_ids,
                                   __global int *entity_flags,
                                   __global int *counters,
                                   __global int2 *query,
                                   __global int *total,
                                   int count,
                                   int max_hull)
{
    int hull_id = get_global_id(0);
    if (hull_id >= max_hull) return;
    int mesh_id = hull_mesh_ids[hull_id];
    int hull_flag = hull_flags[hull_id];
    int entity_id = hull_entity_ids[hull_id];
    int entity_flag = entity_flags[entity_id];
    bool out_of_bounds = (hull_flag & OUT_OF_BOUNDS) !=0;
    bool in_perimeter = (hull_flag & IN_PERIMETER) !=0;
    bool is_ghost = (hull_flag & GHOST_HULL) !=0;
    bool is_active = (entity_flag & GHOST_ACTIVE) !=0;
    bool skip_ghost = is_ghost && !is_active;
    if (out_of_bounds || in_perimeter || skip_ghost) return;

    for (int i = 0; i < count; i++)
    {
        int nx = query[i].x;
        if (mesh_id == nx)
        {
            int x = atomic_inc(&counters[i]);
            int y = atomic_inc(&total[0]);
        }
    }
}

__kernel void write_mesh_details(__global int *hull_mesh_ids,
                                 __global int *hull_flags,
                                 __global int *hull_entity_ids,
                                 __global int *entity_flags,
                                 __global int2 *mesh_vertex_tables,
                                 __global int2 *mesh_face_tables,
                                 __global int *counters, 
                                 __global int2 *query, 
                                 __global int *offsets,
                                 __global int4 *mesh_details,
                                 __global int *mesh_texture,
                                 int count,
                                 int max_hull)
{
    int hull_id = get_global_id(0);
    if (hull_id >= max_hull) return;
    int mesh_id = hull_mesh_ids[hull_id];
    int hull_flag = hull_flags[hull_id];
    int entity_id = hull_entity_ids[hull_id];
    int entity_flag = entity_flags[entity_id];
    bool out_of_bounds = (hull_flag & OUT_OF_BOUNDS) !=0;
    bool in_perimeter = (hull_flag & IN_PERIMETER) !=0;
    bool is_ghost = (hull_flag & GHOST_HULL) !=0;
    bool is_active = (entity_flag & GHOST_ACTIVE) !=0;
    bool skip_ghost = is_ghost && !is_active;
    if (out_of_bounds || in_perimeter || skip_ghost) return;

    for (int i = 0; i < count; i++)
    {
        int2 q = query[i];
        int nx = q.x;
        if (mesh_id == nx)
        {
            int2 mesh_vertex_table = mesh_vertex_tables[mesh_id];
            int2 mesh_face_table = mesh_face_tables[mesh_id];
            int offset = offsets[i];
            int bank = atomic_dec(&counters[i]) - 1;
            int4 out;
            out.x = mesh_vertex_table.y - mesh_vertex_table.x + 1;
            out.y = (mesh_face_table.y - mesh_face_table.x + 1) * 3;
            out.z = hull_id;
            int id = offset + bank;
            mesh_details[id] = out;
            mesh_texture[id] = q.y;
        }
    }
}

__kernel void count_mesh_batches(__global int4 *mesh_details, 
                                 __global int *total,
                                 int max_per_batch,
                                 int count)
{
    int current_batch_count = 0;
    int current_batch = 0;
    for (int i = 0; i < count; i++)
    {
        int4 next = mesh_details[i];
        int next_e_total = next.y + current_batch_count;
        if (next_e_total > max_per_batch)
        {
            current_batch++;
            current_batch_count = 0;
        }
        next.w = current_batch;
        mesh_details[i] = next;
        current_batch_count += next.y;
    }
    int bc = current_batch + 1;
    total[0] = bc;
}

__kernel void calculate_batch_offsets(__global int *mesh_offsets,
                                      __global int4 *mesh_details, 
                                      int count)
{
    int current_batch = 0;
    int last_new = -1;
    for (int i = 0; i < count; i++)
    {
        int4 next = mesh_details[i];
        if (next.w != last_new)
        {
            last_new = next.w;
            mesh_offsets[current_batch++] = i;
        }
    }
}

__kernel void transfer_detail_data(__global int4 *mesh_details, 
                                   __global int2 *mesh_transfer,
                                   int offset,
                                   int max_mesh)
{
    int t_index = get_global_id(0);
    if (t_index >= max_mesh) return;
    int d_index = t_index + offset;
    int2 details = mesh_details[d_index].xy;
    mesh_transfer[t_index] = details;
}


inline float map(float x, float in_min, float in_max, float out_min, float out_max)
{
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

__kernel void transfer_render_data(__global int2 *hull_point_tables,
                                   __global int *hull_mesh_ids,
                                   __global int *hull_entity_ids,
                                   __global int *hull_flags,
                                   __global int *hull_uv_offsets,
                                   __global int *hull_integrity,
                                   __global int *entity_flags,
                                   __global int2 *mesh_vertex_tables,
                                   __global int2 *mesh_face_tables,
                                   __global int4 *mesh_faces,
                                   __global float4 *points,
                                   __global short *point_hit_counts,
                                   __global int *point_vertex_references,
                                   __global int2 *uv_tables,
                                   __global float2 *texture_uvs,
                                   __global int *command_buffer,
                                   __global float4 *vertex_buffer,
                                   __global float2 *uv_buffer,
                                   __global float4 *color_buffer,
                                   __global float *slot_buffer,
                                   __global int *element_buffer,
                                   __global int4 *mesh_details,
                                   __global int *mesh_texture,
                                   __global int2 *mesh_transfer,
                                   int offset,
                                   int max_index)
{
    int t_index = get_global_id(0);
    if (t_index >= max_index) return;
    int d_index = t_index + offset;
    int c_index = t_index * 5;
    int4 details = mesh_details[d_index];
    int texture = mesh_texture[d_index];
    int2 transfer = mesh_transfer[t_index];

    command_buffer[c_index]     = details.y;
    command_buffer[c_index + 1] = 1;
    command_buffer[c_index + 2] = transfer.y;
    command_buffer[c_index + 3] = transfer.x;
    command_buffer[c_index + 4] = t_index;

    int hull_id = details.z;
    int flags = hull_flags[hull_id];
    int mesh_id = hull_mesh_ids[hull_id];
    int2 point_table = hull_point_tables[hull_id];
    int2 mesh_vertex_table = mesh_vertex_tables[mesh_id];
    int2 mesh_face_table = mesh_face_tables[mesh_id];
    int entity_id = hull_entity_ids[hull_id];
    int uv_offset = hull_uv_offsets[hull_id];
    int integrity = hull_integrity[hull_id];
    int a_flags = entity_flags[entity_id];

    bool face_l = (a_flags & FACE_LEFT) !=0;
    bool side_r = (flags & SIDE_R) !=0;
    bool side_l = (flags & SIDE_L) !=0;
    bool touch_alike = (flags & TOUCH_ALIKE) !=0;
    bool is_static = (flags & IS_STATIC) !=0;
    bool cursor_over = (flags & CURSOR_OVER) !=0;
    bool in_range = (flags & IN_RANGE) !=0;
    bool cursor_hit = (flags & CURSOR_HIT) !=0;

    float r_layer = face_l ? -2.0 : 2.0;
    float l_layer = face_l ? 2.0 : -2.0;

    int wrappedValue = hull_id % 100000;
    float normalized = native_divide((float) wrappedValue, (float) (100000 - 1));
    float mappedValue = normalized * 0.99f;

    float side_z = side_r 
        ? r_layer 
        : side_l 
            ? l_layer 
            : mappedValue; 

    int loop_index = 0;

    __attribute__((opencl_unroll_hint(4)))
    for (loop_index = point_table.x; loop_index <= point_table.y; loop_index++)
    {
        int point_vertex_reference = point_vertex_references[loop_index];
        int ref_offset = point_vertex_reference - mesh_vertex_table.x + transfer.x;
        vertex_buffer[ref_offset] = (float4)(points[loop_index].xy, side_z, 1.0f);
        uv_buffer[ref_offset] = texture_uvs[uv_tables[point_vertex_reference].x + uv_offset];

        int hit_counts = point_hit_counts[loop_index];

        float hit_color = map((float) hit_counts, 0, HIT_TOP_THRESHOLD, 0.0f, 0.5f);

        float col = hit_counts <= HIT_LOW_THRESHOLD 
            ? 1.0f 
            : 1 - hit_color;

        float aaa = integrity > 100 
            ? 0.0f 
            : 1.0f - map((float) integrity, 0.0f, 100.0f, 0.0f, 1.0f);

        aaa = 1.0 - aaa;
        
        float rrr  = cursor_hit && !in_range 
            ? 0.5f 
            : 0;

        float bbb = cursor_over && in_range 
            ? 3.0f 
            : 0.0f;

        float ggg = cursor_hit && in_range
            ? 1.0f 
            : 0.0f;

        //if (cursor_hit) printf("debug: hit  over: %d  in_range: %d", cursor_over, in_range);
        color_buffer[ref_offset] = (float4)((col + rrr) * aaa, (col + ggg) * aaa, (col + bbb - ggg * 3) * aaa, 1.0f);
        slot_buffer[ref_offset] = (float)texture;
    }

    __attribute__((opencl_unroll_hint(2)))
    for (loop_index = mesh_face_table.x; loop_index <= mesh_face_table.y; loop_index++)
    {
        int4 face = mesh_faces[loop_index];
        int o1 = (loop_index - mesh_face_table.x) * 3 + transfer.y;
        element_buffer[o1] = face.x;
        element_buffer[o1 + 1] = face.y;
        element_buffer[o1 + 2] = face.z;
    }
}
