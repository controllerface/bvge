typedef struct 
{
    int edge_count;
    int bone_count;
    int point_count;
    int hull_count;
    int armature_count;
} DropCounts;

inline DropCounts calculate_drop_counts(int armature_id,
                                        __global int4 *armature_flags,
                                        __global int2 *hull_tables,
                                        __global int4 *element_tables)
{
    DropCounts drop_counts;
    drop_counts.bone_count = 0;
    drop_counts.point_count = 0;
    drop_counts.edge_count = 0;
    drop_counts.hull_count = 0;
    drop_counts.armature_count = 0;

    int4 armature_flag = armature_flags[armature_id];
    bool deleted = (armature_flag.z & OUT_OF_BOUNDS) !=0;
            
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

__kernel void locate_out_of_bounds(__global int2 *hull_tables,
                                   __global int2 *hull_flags,
                                   __global int4 *armature_flags)
{
    int gid = get_global_id(0);
    int2 hull_table = hull_tables[gid];
    int hull_count = hull_table.y - hull_table.x + 1;
    
    int out_count = 0;
    for (int i = 0; i < hull_count; i++)
    {
        int current_hull = hull_table.x + i;
        int2 hull_flag = hull_flags[current_hull];
        bool is_out = (hull_flag.x & OUT_OF_BOUNDS) !=0;
        if (is_out)
        {
            out_count++;
        }
    }

    if (out_count == hull_count)
    {
        int4 armature_flag = armature_flags[gid];
        int z = armature_flag.z;
        z = (z | OUT_OF_BOUNDS);
        armature_flags[gid].z = z;
    }
}

__kernel void scan_deletes_single_block_out(__global int4 *armature_flags,
                                            __global int2 *hull_tables,
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

    DropCounts a_counts = calculate_drop_counts(a_index, armature_flags, hull_tables, element_tables);
    DropCounts b_counts = calculate_drop_counts(b_index, armature_flags, hull_tables, element_tables);

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

__kernel void scan_deletes_multi_block_out(__global int4 *armature_flags,
                                           __global int2 *hull_tables,
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

    DropCounts a_counts = calculate_drop_counts(a_index, armature_flags, hull_tables, element_tables);
    DropCounts b_counts = calculate_drop_counts(b_index, armature_flags, hull_tables, element_tables);

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

__kernel void complete_deletes_multi_block_out(__global int4 *armature_flags,
                                               __global int2 *hull_tables,
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
            DropCounts a_counts = calculate_drop_counts(a_index, armature_flags, hull_tables, element_tables);
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
            DropCounts b_counts = calculate_drop_counts(b_index, armature_flags, hull_tables, element_tables);
            sz[0] = (output[b_index] + b_counts.edge_count);
            sz[1] = (output2[b_index].x + b_counts.bone_count);
            sz[2] = (output2[b_index].y + b_counts.point_count);
            sz[3] = (output2[b_index].z + b_counts.hull_count);
            sz[4] = (output2[b_index].w + b_counts.armature_count);
        }
    }
}


__kernel void perform_deletes(__global int *buffer_in,
                              __global int4 *buffer_in_2,
                              __global float4 *armatures,
                              __global int4 *armature_flags,
                              __global int2 *hull_tables,
                              __global float4 *hulls,
                              __global int2 *hull_flags,
                              __global float2 *hull_rotations,
                              __global int4 *element_tables,
                              __global float16 *bone_instances,
                              __global int *bone_indices,
                              __global float4 *points,
                              __global float *point_anti_grav,
                              __global int2 *vertex_tables,
                              __global float4 *edges)
{
    // get drop counts for this armature
    int gid = get_global_id(0);
    int buffer_1 = buffer_in[gid];
    int4 buffer_2 = buffer_in_2[gid];

    // todo: check armature for deleted flag, if true do nothing. Adjustments are
    //  only valid for items that are not being removed, their data will overwrite
    //  the data of the deleted objects.

    DropCounts drop;
    drop.edge_count = buffer_1;
    drop.bone_count = buffer_2.x;
    drop.point_count = buffer_2.y;
    drop.hull_count = buffer_2.z;
    drop.armature_count = buffer_2.w;

    float4 armature = armatures[gid];
    int4 armature_flag = armature_flags[gid];
    int2 hull_table = hull_tables[gid];

    int new_armature_index = gid - drop.armature_count;
    // for this armature,
    // -move current float4 position data to new index
    armatures[new_armature_index] = armature;

    // update armature flags .x by hull offset
    int4 new_armature_flag = armature_flag;
    new_armature_flag.x -= drop.hull_count;

    // -move new int4 flag data to new index
    armature_flags[new_armature_index] = new_armature_flag;

    // update hull table .x/.y by hull offset
    int2 new_hull_table = hull_table;
    new_hull_table.x -= drop.hull_count;
    new_hull_table.y -= drop.hull_count;

    // -move new int2 hull table data to new index
    hull_tables[new_armature_index] = new_hull_table;

    // loop current hulls,
    int hull_count = hull_table.y - hull_table.x + 1;
    for (int i = 0; i < hull_count; i++)
    {
        int current_hull = hull_table.x + i;
        float4 hull = hulls[current_hull];
        float2 hull_rotation = hull_rotations[current_hull];
        int2 hull_flag = hull_flags[current_hull];
        int4 element_table = element_tables[current_hull];

        int new_hull_index = current_hull - drop.hull_count;

        //  -move current float4 position data to new index
        hulls[new_hull_index] = hull;

        //  -move current float2 rotation data to new index
        hull_rotations[new_hull_index] = hull_rotation;

        //  update hull flags.y by armature offset
        int2 new_hull_flag = hull_flag;
        new_hull_flag.y -= drop.armature_count;

        //  -move new int2 flag data to new index
        hull_flags[new_hull_index] = new_hull_flag;

        //  update element table .x/.y by point offset
        //  update element table .z/.w by edges offset
        int4 new_element_table = element_table;
        new_element_table.x -= drop.point_count;
        new_element_table.y -= drop.point_count;
        new_element_table.z -= drop.edge_count;
        new_element_table.w -= drop.edge_count;
        
        //  -move new int4 element data to new index
        element_tables[new_hull_index] = new_element_table;
        
        int point_count = element_table.y - element_table.x + 1;
        int edge_count = element_table.w - element_table.z + 1;

        //  loop current edges,
        //   update edge .x/.y by point offset
        //   -move new float4 edge data to new index
        //  loop current points,
        //   -move current float4 position data to new index
        //   -move current float anti-grav data to new index
        //   update vertex table .y by bone offset
        //   -move new int2 vertex table data to new index
        //   get current bone,
        //    -move current float16 matrix data to new index
        //    -move current int bone ref data to new index
    }


}