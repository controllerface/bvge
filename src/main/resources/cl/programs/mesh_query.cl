__kernel void count_mesh_instances(__global int *hull_mesh_ids,
                                   __global int *counters,
                                   __global int *query,
                                   __global int *total,
                                   int count)
{
    int hull_id = get_global_id(0);
    int mesh_id = hull_mesh_ids[hull_id];
    for (int i = 0; i < count; i++)
    {
        int nx = query[i];
        if (mesh_id == nx)
        {
            int x = atomic_inc(&counters[i]);
            int y = atomic_inc(&total[0]);
        }
    }
}

__kernel void write_mesh_details(__global int *hull_mesh_ids,
                                 __global int2 *mesh_vertex_tables,
                                 __global int2 *mesh_face_tables,
                                 __global int *counters, 
                                 __global int *query, 
                                 __global int *offsets,
                                 __global int4 *mesh_details,
                                 int count)
{
    int hull_id = get_global_id(0);
    int mesh_id = hull_mesh_ids[hull_id];
    for (int i = 0; i < count; i++)
    {
        int nx = query[i];
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
                                   int offset)
{
    int t_index = get_global_id(0);
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
                                   __global int *hull_armature_ids,
                                   __global int *hull_flags,
                                   __global int *hull_uv_offsets,
                                   __global int *hull_integrity,
                                   __global int *armature_flags,
                                   __global int2 *mesh_vertex_tables,
                                   __global int2 *mesh_face_tables,
                                   __global int4 *mesh_faces,
                                   __global float4 *points,
                                   __global ushort *point_hit_counts,
                                   __global int *point_vertex_references,
                                   __global int2 *uv_tables,
                                   __global float2 *texture_uvs,
                                   __global int *command_buffer,
                                   __global float4 *vertex_buffer,
                                   __global float2 *uv_buffer,
                                   __global float4 *color_buffer,
                                   __global int *element_buffer,
                                   __global int4 *mesh_details,
                                   __global int2 *mesh_transfer,
                                   int offset)
{
    int t_index = get_global_id(0);
    int d_index = t_index + offset;
    int c_index = t_index * 5;
    int4 details = mesh_details[d_index];
    int2 transfer = mesh_transfer[t_index];

    command_buffer[c_index] = details.y;
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
    int armature_id = hull_armature_ids[hull_id];
    int uv_offset = hull_uv_offsets[hull_id];
    int integrity = hull_integrity[hull_id];
    int a_flags = armature_flags[armature_id];

    bool face_l = (a_flags & FACE_LEFT) !=0;
    bool side_r = (flags & SIDE_R) !=0;
    bool side_l = (flags & SIDE_L) !=0;
    bool touch_alike = (flags & TOUCH_ALIKE) !=0;
    bool is_static = (flags & IS_STATIC) !=0;

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

    int start_point = point_table.x;
    int end_point = point_table.y;
    for (int point_id = start_point; point_id <= end_point; point_id++)
    {
        float4 point = points[point_id];
        int hit_counts = point_hit_counts[point_id];

        float col = hit_counts <= HIT_LOW_THRESHOLD 
            ? 1.0f 
            : hit_counts <= HIT_LOW_MID_THRESHOLD 
                ? 0.80f 
                : hit_counts <= HIT_MID_THRESHOLD
                    ? 0.60f
                    : hit_counts <= HIT_HIGH_MID_THRESHOLD 
                        ? 0.55
                        : 0.50;

        int point_vertex_reference = point_vertex_references[point_id];
        int2 uv_table = uv_tables[point_vertex_reference];
        int uv_count = uv_table.y - uv_table.x + 1;
        int uv_index =  uv_table.x + uv_offset;
        float2 uv = texture_uvs[uv_index];
        float4 pos = (float4)(point.xy, side_z, 1.0f);
        int ref_offset = point_vertex_reference - mesh_vertex_table.x + transfer.x;

        float xxx = is_static ? col - 0.07f : col;
        float rrr = integrity > 100 
            ? 0.0f 
            : .3f - map((float) integrity, 0.0f, 100.0f, 0.0f, .3f);
        vertex_buffer[ref_offset] = pos;
        uv_buffer[ref_offset] = uv;
        color_buffer[ref_offset] = (float4)(xxx + rrr, xxx, xxx, 1.0f);
    }

    int start_face = mesh_face_table.x;
    int end_face = mesh_face_table.y;
    for (int face_id = start_face; face_id <= end_face; face_id++)
    {
        int4 face = mesh_faces[face_id];
        int base_offset = face_id - start_face;
        int inner_offset = base_offset * 3;
        int o1 = inner_offset + transfer.y;
        int o2 = o1 + 1;
        int o3 = o1 + 2;
        int p1 =  face.x;
        int p2 =  face.y;
        int p3 =  face.z;
        element_buffer[o1] = p1;
        element_buffer[o2] = p2;
        element_buffer[o3] = p3;
    }
}
