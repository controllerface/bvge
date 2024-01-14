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

__kernel void write_mesh_data(__global int *hull_mesh_ids,
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

__kernel void calculate_batch_offsets(__global int4 *mesh_offsets,
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
            //printf("debug: batch=%d offset=%d", current_batch-1, i);
        }
    }
}