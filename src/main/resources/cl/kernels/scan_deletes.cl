typedef struct 
{
    int edge_count;
    int bone_count;
    int point_count;
    int hull_count;
    int armature_count;
} DropCounts;

inline DropCounts calculate_drop_counts(int armature_id,
                                        __global int2 *hull_tables,
                                        __global int4 *element_tables)
{
    DropCounts drop_counts;
    drop_counts.bone_count = 0;
    drop_counts.point_count = 0;
    drop_counts.edge_count = 0;
    drop_counts.hull_count = 0;
    drop_counts.armature_count = 0;

    // todo: needed deleted flag in armature data
    bool deleted = true;
            
    if (deleted)
    {
        drop_counts.armature_count = 1;
        int2 hull_table = hull_tables[armature_id];
        int hull_count = hull_table.y - hull_table.x + 1;
        drop_counts.hull_count = hull_count;
        drop_counts.bone_count = hull_count; // todo: this is true for now, but may change
        for (int i = 0; i < hull_count; i++)
        {
            int current_hull = hull_table.x + i;
            int4 element_table = element_tables[current_hull];
            int point_count = element_table.y - element_table.x + 1;
            int edge_count = element_table.w - element_table.z + 1;
            drop_counts.point_count += point_count;
            drop_counts.edge_count += edge_count;
        }
    }
    return drop_counts;
}

// output buffer layout:
// - int : edge count
// - int4 x: bone count
//        y: point count
//        z: hull count
//        w: armature count
//
__kernel void scan_deletes_single_block_out(__global int2 *hull_tables,
                                            __global int4 *element_tables,
                                            __global int *output,
                                            __global int4 *output2,
                                            __global int *sz,
                                            __local int *buffer, 
                                            __local int4 *buffer2, 
                                            int n) 
{
    int global_id = get_global_id(0);

    int a_index = (global_id * 2);
    int b_index = (global_id * 2) + 1;

    DropCounts a_counts = calculate_drop_counts(a_index, hull_tables, element_tables);
    DropCounts b_counts = calculate_drop_counts(b_index, hull_tables, element_tables);

    int m = 2 * get_local_size(0);

    if (a_index < n)
    {
         buffer[a_index] = a_counts.edge_count;
         buffer2[a_index].x = a_counts.bone_count;
         buffer2[a_index].y = a_counts.point_count;
         buffer2[a_index].z = a_counts.hull_count;
         buffer2[a_index].w = a_counts.armature_count;
    }
    else 
    {
        buffer[a_index] = 0;
        buffer2[a_index].x = 0;
        buffer2[a_index].y = 0;
        buffer2[a_index].z = 0;
        buffer2[a_index].w = 0;
    }

    if (b_index < n)
    {
         buffer[b_index] = b_counts.edge_count;
         buffer2[b_index].x = b_counts.bone_count;
         buffer2[b_index].y = b_counts.point_count;
         buffer2[b_index].z = b_counts.hull_count;
         buffer2[b_index].w = b_counts.armature_count;
    }
    else 
    {
        buffer[b_index] = 0;
        buffer2[b_index].x = 0;
        buffer2[b_index].y = 0;
        buffer2[b_index].z = 0;
        buffer2[b_index].w = 0;
    }

    upsweep_ex(buffer, buffer2, m);

    if (b_index == (m - 1)) 
    {
        buffer[b_index] = 0;
        buffer2[b_index].x = 0;
        buffer2[b_index].y = 0;
        buffer2[b_index].z = 0;
        buffer2[b_index].w = 0;
    }

    downsweep_ex(buffer, buffer2, m);

    if (a_index < n) 
    {
        output[a_index] = buffer[a_index];
        output2[a_index] = buffer2[a_index];
        if (a_index == n - 1)
        {
            sz[0] = (output[a_index] + a_counts.edge_count);
            sz[1] = (output2[a_index].x + a_counts.bone_count);
            sz[2] = (output2[a_index].y + a_counts.point_count);
            sz[3] = (output2[a_index].z + a_counts.hull_count);
            sz[4] = (output2[a_index].w + a_counts.armature_count);
        }
    }

    if (b_index < n) 
    {
        output[b_index] = buffer[b_index];
        output2[b_index] = buffer2[b_index];
        if (b_index == n - 1)
        {
            sz[0] = (output[b_index] + b_counts.edge_count);
            sz[1] = (output2[b_index].x + b_counts.bone_count);
            sz[2] = (output2[b_index].y + b_counts.point_count);
            sz[3] = (output2[b_index].z + b_counts.hull_count);
            sz[4] = (output2[b_index].w + b_counts.armature_count);
        }
    }
}

__kernel void scan_deletes_multi_block_out(__global int2 *hull_tables,
                                           __global int4 *element_tables,
                                           __global int *output,
                                           __global int4 *output2,
                                           __local int *buffer, 
                                           __local int4 *buffer2, 
                                           __global int *part, 
                                           __global int4 *part2, 
                                           int n)
{
    int wx = get_local_size(0);

    int global_id = get_global_id(0);
    int a_index = (2 * global_id);
    int b_index = (2 * global_id) + 1;

    DropCounts a_counts = calculate_drop_counts(a_index, hull_tables, element_tables);
    DropCounts b_counts = calculate_drop_counts(b_index, hull_tables, element_tables);

    int local_id = get_local_id(0);
    int local_a_index = (2 * local_id);
    int local_b_index = (2 * local_id) + 1;
    int grpid = get_group_id(0);

    int m = wx * 2;
    int k = get_num_groups(0);

    if (a_index < n)
    {
         buffer[local_a_index] = a_counts.edge_count;
         buffer2[local_a_index].x = a_counts.bone_count;
         buffer2[local_a_index].y = a_counts.point_count;
         buffer2[local_a_index].z = a_counts.hull_count;
         buffer2[local_a_index].w = a_counts.armature_count;
    }
    else 
    {
        buffer[local_a_index] = 0;
        buffer2[local_a_index].x = 0;
        buffer2[local_a_index].y = 0;
        buffer2[local_a_index].z = 0;
        buffer2[local_a_index].w = 0;
    }

    if (b_index < n)
    {
         buffer[local_b_index] = b_counts.edge_count;
         buffer2[local_b_index].x = b_counts.bone_count;
         buffer2[local_b_index].y = b_counts.point_count;
         buffer2[local_b_index].z = b_counts.hull_count;
         buffer2[local_b_index].w = b_counts.armature_count;
    }
    else 
    {
        buffer[local_b_index] = 0;
        buffer2[local_b_index] = (int4)(0, 0, 0, 0);
    }

    upsweep_ex(buffer, buffer2, m);

    if (local_id == (wx - 1)) 
    {
        part[grpid] = buffer[local_b_index];
        part2[grpid] = buffer2[local_b_index];
        buffer[local_b_index] = 0;
        buffer2[local_b_index] = (int4)(0, 0, 0, 0);
    }

    downsweep_ex(buffer, buffer2, m);

    if (a_index < n) 
    {
        output[a_index] = buffer[local_a_index];
        output2[a_index] = buffer2[local_a_index];
    }
    if (b_index < n) 
    {
        output[b_index] = buffer[local_b_index];
        output2[b_index] = buffer2[local_b_index];
    }
}

__kernel void complete_deletes_multi_block_out(__global int2 *hull_tables,
                                               __global int4 *element_tables,
                                               __global int *output,
                                               __global int4 *output2,
                                               __global int *sz,
                                               __local int *buffer, 
                                               __local int4 *buffer2,
                                               __global int *part, 
                                               __global int4 *part2,
                                               int n)
{
    int global_id = get_global_id(0);
    int a_index = (2 * global_id);
    int b_index = (2 * global_id) + 1;

    int local_id = get_local_id(0);
    int local_a_index = (2 * local_id);
    int local_b_index = (2 * local_id) + 1;
    int grpid = get_group_id(0);

    if (a_index < n)
    {
        buffer[local_a_index] = output[a_index];
        buffer2[local_a_index] = output2[a_index];
    }
    else
    {
        buffer[local_a_index] = 0;
        buffer2[local_a_index] = (int4)(0, 0, 0, 0);
    }

    if (b_index < n)
    {
        buffer[local_b_index] = output[b_index];
        buffer2[local_b_index] = output2[b_index];
    }
    else
    {
        buffer[local_b_index] = 0;
        buffer2[local_b_index] = (int4)(0, 0, 0, 0);
    }


    buffer[local_a_index] += part[grpid];
    buffer[local_b_index] += part[grpid];
    buffer2[local_a_index] += part2[grpid];
    buffer2[local_b_index] += part2[grpid];


    if (a_index < n) 
    {
        output[a_index] = buffer[local_a_index];
        output2[a_index] = buffer2[local_a_index];
        if (a_index == n - 1)
        {
            DropCounts a_counts = calculate_drop_counts(a_index, hull_tables, element_tables);
            sz[0] = (output[a_index] + a_counts.edge_count);
            sz[1] = (output2[a_index].x + a_counts.bone_count);
            sz[2] = (output2[a_index].y + a_counts.point_count);
            sz[3] = (output2[a_index].z + a_counts.hull_count);
            sz[4] = (output2[a_index].w + a_counts.armature_count);
        }
    }
    if (b_index < n) 
    {
        output[b_index] = buffer[local_b_index];
        output2[b_index] = buffer2[local_b_index];
        if (b_index == n - 1)
        {
            DropCounts b_counts = calculate_drop_counts(b_index, hull_tables, element_tables);
            sz[0] = (output[b_index] + b_counts.edge_count);
            sz[1] = (output2[b_index].x + b_counts.bone_count);
            sz[2] = (output2[b_index].y + b_counts.point_count);
            sz[3] = (output2[b_index].z + b_counts.hull_count);
            sz[4] = (output2[b_index].w + b_counts.armature_count);
        }
    }
}
