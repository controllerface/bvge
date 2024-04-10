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

__kernel void transfer_render_data(__global int4 *hull_element_tables,
                                   __global int *hull_mesh_ids,
                                   __global int2 *mesh_vertex_tables,
                                   __global int2 *mesh_face_tables,
                                   __global int4 *mesh_faces,
                                   __global float4 *points,
                                   __global int *point_vertex_references,
                                   __global int2 *uv_tables,
                                   __global float2 *texture_uvs,
                                   __global int *command_buffer,
                                   __global float2 *vertex_buffer,
                                   __global float2 *uv_buffer,
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
    int mesh_id = hull_mesh_ids[hull_id];
    int4 element_table = hull_element_tables[hull_id];
    int2 mesh_vertex_table = mesh_vertex_tables[mesh_id];
    int2 mesh_face_table = mesh_face_tables[mesh_id];

    int start_point = element_table.x;
    int end_point = element_table.y;
    for (int point_id = start_point; point_id <= end_point; point_id++)
    {
        float4 point = points[point_id];
        int point_vertex_reference = point_vertex_references[point_id];
        int2 uv_table = uv_tables[point_vertex_reference];
        float2 uv = texture_uvs[uv_table.x]; // todo: select from available uvs based on hull data
        float2 pos = point.xy;
        int ref_offset = point_vertex_reference - mesh_vertex_table.x + transfer.x;
        vertex_buffer[ref_offset] = pos;
        uv_buffer[ref_offset] = uv;
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