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
            //printf("debug: hull id: %d mesh id: %d count: %d total: %d", hull_id, mesh_id, x+1, y+1);
        }
    }
}

__kernel void write_mesh_details(__global int *hull_mesh_ids,
                                 __global int4 *mesh_references,
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
            int4 ref_mesh = mesh_references[mesh_id];
            int offset = offsets[i];
            int bank = atomic_dec(&counters[i]) - 1;
            int4 out;
            out.x = ref_mesh.y - ref_mesh.x + 1;
            out.y = (ref_mesh.w - ref_mesh.z + 1) * 3;
            out.z = hull_id;
            int id = offset + bank;
            mesh_details[id] = out;
            // printf("debug: hull id: %d mesh id: %d offset: %d bank: %d vc: %d fc: %d", 
            //     hull_id, mesh_id, offset, bank, out.x, out.y);
            //printf("debug: out.x=%d out.y=%d out.z=%d", out.x, out.y, out.z);
        }
    }
}

__kernel void count_mesh_batches(__global int4 *mesh_details, 
                                 __global int *total,
                                 int count)
{
    // todo: this needs to be configurable
    int max_per_batch = 400;
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
    //printf("debug: batches=%d", bc);
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
            //printf("debug: batch=%d offset=%d", current_batch, i);
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
    //printf("counts: x=%d y=%d t_id=%d d_id=%d", details.x, details.y, t_index, d_index);
}

__kernel void transfer_render_data(__global int4 *hull_element_tables,
                                   __global int *hull_mesh_ids,
                                   __global int4 *mesh_references,
                                   __global int4 *mesh_faces,
                                   __global float4 *points,
                                   __global int4 *vertex_tables,
                                   __global int *command_buffer,
                                   __global float2 *vertex_buffer,
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
    int4 mesh_reference = mesh_references[mesh_id];
    int start_point = element_table.x;
    int end_point = element_table.y;
    for (int point_id = start_point; point_id <= end_point; point_id++)
    {
        float4 point = points[point_id];
        int4 vertex_table = vertex_tables[point_id];
        float2 pos = point.xy;
        int ref_offset = vertex_table.x - mesh_reference.x + transfer.x;
        vertex_buffer[ref_offset] = pos;
    }

    int face_offset = 0;
    int start_face = mesh_reference.z;
    int end_face = mesh_reference.w;
    for (int face_id = start_face; face_id <= end_face; face_id++)
    {
        int4 face = mesh_faces[face_id];
        int ref_offset = face_id - start_face + transfer.y + face_offset;
        element_buffer[ref_offset] = face.x;
        element_buffer[ref_offset + 1] = face.y;
        element_buffer[ref_offset + 2] = face.z;
        face_offset += 3;
        printf("face debug: x=%d y=%d z=%d", face.x, face.y, face.z);
    }
}