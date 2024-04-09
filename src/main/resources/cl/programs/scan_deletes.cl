typedef struct 
{
    int edge_count;
    int bone_count;
    int point_count;
    int hull_count;
    int armature_count;
    int bone_bind_count;
} DropCounts;

inline DropCounts calculate_drop_counts(int armature_id,
                                        __global int *armature_flags,
                                        __global int4 *hull_tables,
                                        __global int4 *element_tables,
                                        __global int2 *hull_bone_tables)
{
    // todo: add armature bone counts
    DropCounts drop_counts;
    drop_counts.bone_count = 0;
    drop_counts.point_count = 0;
    drop_counts.edge_count = 0;
    drop_counts.hull_count = 0;
    drop_counts.armature_count = 0;
    drop_counts.bone_bind_count = 0;

    int flags = armature_flags[armature_id];
    bool deleted = (flags & DELETED) !=0;
            
    if (deleted)
    {
        drop_counts.armature_count = 1;
        int4 hull_table = hull_tables[armature_id];
        int hull_count = hull_table.y - hull_table.x + 1;
        int bone_bind_count = hull_table.w - hull_table.z + 1;
        drop_counts.hull_count = hull_count;
        drop_counts.bone_bind_count = bone_bind_count;

        for (int i = 0; i < hull_count; i++)
        {
            int current_hull = hull_table.x + i;
            int4 element_table = element_tables[current_hull];
            int2 bone_table = hull_bone_tables[current_hull];
            int bone_count = bone_table.y - bone_table.x + 1;
            int point_count = element_table.y - element_table.x + 1;
            int edge_count = element_table.w >= 0 
                ? element_table.w - element_table.z + 1 
                : 0;
            drop_counts.bone_count += bone_count;
            drop_counts.point_count += point_count;
            drop_counts.edge_count += edge_count;
        }
    }
    return drop_counts;
}

__kernel void locate_out_of_bounds(__global int4 *hull_tables,
                                   __global int *hull_flags,
                                   __global int *armature_flags,
                                   __global int *counter)
{
    int gid = get_global_id(0);
    int4 hull_table = hull_tables[gid];
    int hull_count = hull_table.y - hull_table.x + 1;
    
    int out_count = 0;
    for (int i = 0; i < hull_count; i++)
    {
        int current_hull = hull_table.x + i;
        int hull_flag = hull_flags[current_hull];
        bool is_out = (hull_flag & OUT_OF_BOUNDS) !=0;
        if (is_out)
        {
            out_count++;
        }
    }

    if (out_count == hull_count)
    {
        int flags = armature_flags[gid];
        flags = (flags | DELETED);
        armature_flags[gid] = flags;
    }
}

__kernel void scan_deletes_single_block_out(__global int *armature_flags,
                                            __global int4 *hull_tables,
                                            __global int4 *element_tables,
                                            __global int2 *hull_bone_tables,
                                            __global int2 *output1,
                                            __global int4 *output2,
                                            __global int *sz,
                                            __local int2 *buffer1,
                                            __local int4 *buffer2, 
                                            int n) 
{
    int global_id = get_global_id(0);

    int a_index = (global_id * 2);
    int b_index = (global_id * 2) + 1;

    DropCounts a_counts = calculate_drop_counts(a_index, armature_flags, hull_tables, element_tables, hull_bone_tables);
    DropCounts b_counts = calculate_drop_counts(b_index, armature_flags, hull_tables, element_tables, hull_bone_tables);

    int m = 2 * get_local_size(0);

    if (a_index < n)
    {
         buffer1[a_index].x = a_counts.edge_count;
         buffer1[a_index].y = a_counts.bone_bind_count;
         buffer2[a_index].x = a_counts.bone_count;
         buffer2[a_index].y = a_counts.point_count;
         buffer2[a_index].z = a_counts.hull_count;
         buffer2[a_index].w = a_counts.armature_count;
    }
    else 
    {
        buffer1[a_index].x = 0;
        buffer1[a_index].y = 0;
        buffer2[a_index].x = 0;
        buffer2[a_index].y = 0;
        buffer2[a_index].z = 0;
        buffer2[a_index].w = 0;
    }

    if (b_index < n)
    {
         buffer1[b_index].x = b_counts.edge_count;
         buffer1[b_index].y = b_counts.bone_bind_count;
         buffer2[b_index].x = b_counts.bone_count;
         buffer2[b_index].y = b_counts.point_count;
         buffer2[b_index].z = b_counts.hull_count;
         buffer2[b_index].w = b_counts.armature_count;
    }
    else 
    {
        buffer1[b_index].x = 0;
        buffer1[b_index].y = 0;
        buffer2[b_index].x = 0;
        buffer2[b_index].y = 0;
        buffer2[b_index].z = 0;
        buffer2[b_index].w = 0;
    }

    upsweep_ex(buffer1, buffer2, m);

    if (b_index == (m - 1)) 
    {
        buffer1[b_index].x = 0;
        buffer1[b_index].y = 0;
        buffer2[b_index].x = 0;
        buffer2[b_index].y = 0;
        buffer2[b_index].z = 0;
        buffer2[b_index].w = 0;
    }

    downsweep_ex(buffer1, buffer2, m);

    if (a_index < n) 
    {
        output1[a_index] = buffer1[a_index];
        output2[a_index] = buffer2[a_index];
        if (a_index == n - 1)
        {
            sz[0] = (output1[a_index].x + a_counts.edge_count);
            sz[1] = (output2[a_index].x + a_counts.bone_count);
            sz[2] = (output2[a_index].y + a_counts.point_count);
            sz[3] = (output2[a_index].z + a_counts.hull_count);
            sz[4] = (output2[a_index].w + a_counts.armature_count);
            sz[5] = (output1[a_index].y + a_counts.bone_bind_count);
        }
    }

    if (b_index < n) 
    {
        output1[b_index] = buffer1[b_index];
        output2[b_index] = buffer2[b_index];
        if (b_index == n - 1)
        {
            sz[0] = (output1[b_index].x + b_counts.edge_count);
            sz[1] = (output2[b_index].x + b_counts.bone_count);
            sz[2] = (output2[b_index].y + b_counts.point_count);
            sz[3] = (output2[b_index].z + b_counts.hull_count);
            sz[4] = (output2[b_index].w + b_counts.armature_count);
            sz[5] = (output1[b_index].y + b_counts.bone_bind_count);
        }
    }
}

__kernel void scan_deletes_multi_block_out(__global int *armature_flags,
                                           __global int4 *hull_tables,
                                           __global int4 *element_tables,
                                           __global int2 *hull_bone_tables,
                                           __global int2 *output1,
                                           __global int4 *output2,
                                           __local int2 *buffer1, 
                                           __local int4 *buffer2, 
                                           __global int2 *part1, 
                                           __global int4 *part2, 
                                           int n)
{
    int wx = get_local_size(0);

    int global_id = get_global_id(0);
    int a_index = (2 * global_id);
    int b_index = (2 * global_id) + 1;

    DropCounts a_counts = calculate_drop_counts(a_index, armature_flags, hull_tables, element_tables, hull_bone_tables);
    DropCounts b_counts = calculate_drop_counts(b_index, armature_flags, hull_tables, element_tables, hull_bone_tables);

    int local_id = get_local_id(0);
    int local_a_index = (2 * local_id);
    int local_b_index = (2 * local_id) + 1;
    int grpid = get_group_id(0);

    int m = wx * 2;
    int k = get_num_groups(0);

    if (a_index < n)
    {
         buffer1[local_a_index].x = a_counts.edge_count;
         buffer1[local_a_index].y = a_counts.bone_bind_count;
         buffer2[local_a_index].x = a_counts.bone_count;
         buffer2[local_a_index].y = a_counts.point_count;
         buffer2[local_a_index].z = a_counts.hull_count;
         buffer2[local_a_index].w = a_counts.armature_count;
    }
    else 
    {
        buffer1[local_a_index].x = 0;
        buffer1[local_a_index].y = 0;
        buffer2[local_a_index].x = 0;
        buffer2[local_a_index].y = 0;
        buffer2[local_a_index].z = 0;
        buffer2[local_a_index].w = 0;
    }

    if (b_index < n)
    {
         buffer1[local_b_index].x = b_counts.edge_count;
         buffer1[local_b_index].y = b_counts.bone_bind_count;
         buffer2[local_b_index].x = b_counts.bone_count;
         buffer2[local_b_index].y = b_counts.point_count;
         buffer2[local_b_index].z = b_counts.hull_count;
         buffer2[local_b_index].w = b_counts.armature_count;
    }
    else 
    {
        buffer1[local_b_index].x = 0;
        buffer1[local_b_index].y = 0;
        buffer2[local_b_index].x = 0;
        buffer2[local_b_index].y = 0;
        buffer2[local_b_index].z = 0;
        buffer2[local_b_index].w = 0;
    }

    upsweep_ex(buffer1, buffer2, m);

    if (local_id == (wx - 1)) 
    {
        part1[grpid] = buffer1[local_b_index];
        part2[grpid] = buffer2[local_b_index];
        buffer1[local_b_index] = (int2)(0, 0);
        buffer2[local_b_index] = (int4)(0, 0, 0, 0);
    }

    downsweep_ex(buffer1, buffer2, m);

    if (a_index < n) 
    {
        output1[a_index] = buffer1[local_a_index];
        output2[a_index] = buffer2[local_a_index];
    }
    if (b_index < n) 
    {
        output1[b_index] = buffer1[local_b_index];
        output2[b_index] = buffer2[local_b_index];
    }
}

__kernel void complete_deletes_multi_block_out(__global int *armature_flags,
                                               __global int4 *hull_tables,
                                               __global int4 *element_tables,
                                               __global int2 *hull_bone_tables,
                                               __global int2 *output1,
                                               __global int4 *output2,
                                               __global int *sz,
                                               __local int2 *buffer1, 
                                               __local int4 *buffer2,
                                               __global int2 *part1, 
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
        buffer1[local_a_index] = output1[a_index];
        buffer2[local_a_index] = output2[a_index];
    }
    else
    {
        buffer1[local_a_index] = (int2)(0, 0);
        buffer2[local_a_index] = (int4)(0, 0, 0, 0);
    }

    if (b_index < n)
    {
        buffer1[local_b_index] = output1[b_index];
        buffer2[local_b_index] = output2[b_index];
    }
    else
    {
        buffer1[local_b_index] = (int2)(0, 0);
        buffer2[local_b_index] = (int4)(0, 0, 0, 0);
    }


    buffer1[local_a_index] += part1[grpid];
    buffer1[local_b_index] += part1[grpid];
    buffer2[local_a_index] += part2[grpid];
    buffer2[local_b_index] += part2[grpid];


    if (a_index < n) 
    {
        output1[a_index] = buffer1[local_a_index];
        output2[a_index] = buffer2[local_a_index];
        if (a_index == n - 1)
        {
            DropCounts a_counts = calculate_drop_counts(a_index, armature_flags, hull_tables, element_tables, hull_bone_tables);
            sz[0] = (output1[a_index].x + a_counts.edge_count);
            sz[1] = (output2[a_index].x + a_counts.bone_count);
            sz[2] = (output2[a_index].y + a_counts.point_count);
            sz[3] = (output2[a_index].z + a_counts.hull_count);
            sz[4] = (output2[a_index].w + a_counts.armature_count);
            sz[5] = (output1[a_index].y + a_counts.bone_bind_count);
        }
    }
    if (b_index < n) 
    {
        output1[b_index] = buffer1[local_b_index];
        output2[b_index] = buffer2[local_b_index];
        if (b_index == n - 1)
        {
            DropCounts b_counts = calculate_drop_counts(b_index, armature_flags, hull_tables, element_tables, hull_bone_tables);
            sz[0] = (output1[b_index].x + b_counts.edge_count);
            sz[1] = (output2[b_index].x + b_counts.bone_count);
            sz[2] = (output2[b_index].y + b_counts.point_count);
            sz[3] = (output2[b_index].z + b_counts.hull_count);
            sz[4] = (output2[b_index].w + b_counts.armature_count);
            sz[5] = (output1[b_index].y + b_counts.bone_bind_count);
        }
    }
}

__kernel void compact_armatures(__global int2 *buffer_in_1,
                                __global int4 *buffer_in_2,
                                __global float4 *armatures,
                                __global int *armature_root_hulls,
                                __global int *armature_model_indices,
                                __global int *armature_model_transforms,
                                __global int *armature_flags,
                                __global int *armature_animation_indices,
                                __global float *armature_animation_elapsed,
                                __global int4 *hull_tables,
                                __global float4 *hulls,
                                __global int2 *hull_bone_tables,
                                __global int *hull_armature_ids,
                                __global int4 *element_tables,
                                __global float4 *points,
                                __global int *point_hull_indices,
                                __global int4 *bone_tables,
                                __global int2 *bone_bind_tables,
                                __global int *hull_bind_pose_indicies,
                                __global int2 *edges,
                                __global int *bone_shift,
                                __global int *point_shift,
                                __global int *edge_shift,
                                __global int *hull_shift,
                                __global int *bone_bind_shift)
{
    // get drop counts for this armature
    int gid = get_global_id(0);
    int2 buffer_1 = buffer_in_1[gid];
    int4 buffer_2 = buffer_in_2[gid];
    DropCounts drop;
    drop.edge_count = buffer_1.x;
    drop.bone_bind_count = buffer_1.y;
    drop.bone_count = buffer_2.x;
    drop.point_count = buffer_2.y;
    drop.hull_count = buffer_2.z;
    drop.armature_count = buffer_2.w;

    // armature
    float4 armature                 = armatures[gid];
    int armature_root_hull          = armature_root_hulls[gid];
    int armature_model_id           = armature_model_indices[gid];
    int armature_model_transform_id = armature_model_transforms[gid];
    int armature_flag               = armature_flags[gid];
    int4 hull_table                 = hull_tables[gid];
    int anim_index                  = armature_animation_indices[gid];
    float anim_time                 = armature_animation_elapsed[gid];
    
    barrier(CLK_GLOBAL_MEM_FENCE);

    // any armature that is being deleted can be ignored
    bool is_out = (armature_flag & DELETED) !=0;
    if (is_out || drop.armature_count == 0) 
    {
        return;
    }

    // update with drop counts
    int new_armature_index = gid - drop.armature_count;

    int new_armature_root_hull = armature_root_hull;
    new_armature_root_hull -= drop.hull_count;

    int4 new_hull_table = hull_table;
    new_hull_table.x -= drop.hull_count;
    new_hull_table.y -= drop.hull_count;
    new_hull_table.z -= drop.bone_bind_count;
    new_hull_table.w -= drop.bone_bind_count;

    // store updated data at the new index
    armatures[new_armature_index]                  = armature;
    armature_root_hulls[new_armature_index]        = new_armature_root_hull;
    armature_model_indices[new_armature_index]     = armature_model_id;
    armature_model_transforms[new_armature_index]  = armature_model_transform_id;
    armature_flags[new_armature_index]             = armature_flag;
    hull_tables[new_armature_index]                = new_hull_table;
    armature_animation_indices[new_armature_index] = anim_index;
    armature_animation_elapsed[new_armature_index] = anim_time;

    // Note: hull, point, edge, and bone data may be adjusted, but the buffers are not
    // compacted immediately. The offset each object would be moved by, is stored 
    // in an object aliged "shift buffer". Subsequent kernels are then called with the 
    // shift buffers to perform the compaction.

    int armature_bone_count = hull_table.w - hull_table.z + 1;
    for (int i = 0; i < armature_bone_count; i++)
    {
        int current_bone_bind = hull_table.z + i;
        int2 bone_bind_table = bone_bind_tables[current_bone_bind];
        bone_bind_table.y = bone_bind_table.y == -1
            ? -1
            : bone_bind_table.y - drop.bone_bind_count;
        bone_bind_tables[current_bone_bind] = bone_bind_table;
        bone_bind_shift[current_bone_bind] = drop.bone_bind_count;
    }

    // hulls
    int hull_count = hull_table.y - hull_table.x + 1;
    for (int i = 0; i < hull_count; i++)
    {
        int current_hull = hull_table.x + i;

        int4 element_table = element_tables[current_hull];
        int2 hull_bone_table = hull_bone_tables[current_hull];
        int hull_armature_id = hull_armature_ids[current_hull];

        int4 new_element_table = element_table;
        int2 new_hull_bone_table = hull_bone_table;
        int new_hull_armature_id = hull_armature_id;

        new_hull_armature_id = hull_armature_id - drop.armature_count;
        new_hull_bone_table.x -= drop.bone_count;
        new_hull_bone_table.y -= drop.bone_count;
        new_element_table.x -= drop.point_count;
        new_element_table.y -= drop.point_count;
        new_element_table.z -= drop.edge_count;
        new_element_table.w -= drop.edge_count;
        hull_bone_tables[current_hull] = new_hull_bone_table;
        hull_armature_ids[current_hull] = new_hull_armature_id;
        element_tables[current_hull] = new_element_table;
        hull_shift[current_hull] = drop.hull_count;
        
        // edges
        int edge_count = element_table.w - element_table.z + 1;
        for (int j = 0; j < edge_count; j++)
        {
            int current_edge = element_table.z + j;
            int2 edge = edges[current_edge];
            edge.x -= drop.point_count;
            edge.y -= drop.point_count;
            edges[current_edge] = edge;
            edge_shift[current_edge] = drop.edge_count;
        }

        // points
        int point_count = element_table.y - element_table.x + 1;
        for (int k = 0; k < point_count; k++)
        {
            int current_point = element_table.x + k;
            int point_hull_index = point_hull_indices[current_point];
            int4 bone_table = bone_tables[current_point];
            point_hull_index -= drop.hull_count;
            bone_table.x -= bone_table.x > -1 ? drop.bone_count : 0; 
            bone_table.y -= bone_table.y > -1 ? drop.bone_count : 0; 
            bone_table.z -= bone_table.z > -1 ? drop.bone_count : 0; 
            bone_table.w -= bone_table.w > -1 ? drop.bone_count : 0; 
            point_hull_indices[current_point] = point_hull_index;
            bone_tables[current_point] = bone_table;
            point_shift[current_point] = drop.point_count;
        }

        // bones
        int bone_count = hull_bone_table.y - hull_bone_table.x + 1;
        for (int l = 0; l < bone_count; l++)
        {
            int current_bone = hull_bone_table.x + l;
            int hull_bind_pose_index = hull_bind_pose_indicies[current_bone];
            hull_bind_pose_index -= drop.bone_bind_count;
            hull_bind_pose_indicies[current_bone] = hull_bind_pose_index;
            bone_shift[current_bone] = drop.bone_count;
        }
    }
}

__kernel void compact_hulls(__global int *hull_shift,
                            __global float4 *hulls,
                            __global int *hull_mesh_ids,
                            __global float2 *hull_rotations,
                            __global float2 *hull_frictions,
                            __global int2 *bone_tables,
                            __global int *armature_ids,
                            __global int *hull_flags,
                            __global int4 *element_tables,
                            __global float4 *bounds,
                            __global int4 *bounds_index_data,
                            __global int2 *bounds_bank_data)
{
    int current_hull = get_global_id(0);
    int shift = hull_shift[current_hull];
    float4 hull = hulls[current_hull];
    float2 rotation = hull_rotations[current_hull];
    float2 friction = hull_frictions[current_hull];
    int2 bone_table = bone_tables[current_hull];
    int armature_id = armature_ids[current_hull];
    int hull_flag = hull_flags[current_hull];
    int4 element_table = element_tables[current_hull];
    float4 bound = bounds[current_hull];
    int4 bounds_index = bounds_index_data[current_hull];
    int2 bounds_bank = bounds_bank_data[current_hull];
    int hull_mesh_id = hull_mesh_ids[current_hull];
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (shift > 0)
    {
        int new_hull_index = current_hull - shift;

        hulls[new_hull_index] = hull;
        hull_rotations[new_hull_index] = rotation;
        hull_frictions[new_hull_index] = friction;
        bone_tables[new_hull_index] = bone_table;
        armature_ids[new_hull_index] = armature_id;
        hull_flags[new_hull_index] = hull_flag;
        element_tables[new_hull_index] = element_table;
        bounds[new_hull_index] = bound;
        bounds_index_data[new_hull_index] = bounds_index;
        bounds_bank_data[new_hull_index] = bounds_bank;
        hull_mesh_ids[new_hull_index] = hull_mesh_id;
    }
}

__kernel void compact_edges(__global int *edge_shift,
                            __global int2 *edges,
                            __global float *edge_lengths,
                            __global int *edge_flags)
{
    int current_edge = get_global_id(0);
    int shift = edge_shift[current_edge];
    int2 edge = edges[current_edge];
    float edge_length = edge_lengths[current_edge];
    int edge_flag = edge_flags[current_edge];
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (shift > 0)
    {
        int new_edge_index = current_edge - shift;
        edges[new_edge_index] = edge;
        edge_lengths[new_edge_index] = edge_length;
        edge_flags[new_edge_index] = edge_flag;
    }
}

__kernel void compact_points(__global int *point_shift,
                             __global float4 *points,
                             __global float *anti_gravity,
                             __global int *point_vertex_references,
                             __global int *point_hull_indices,
                             __global int *point_flags,
                             __global int4 *bone_tables)
{
    int current_point = get_global_id(0);
    int shift = point_shift[current_point];
    float4 point = points[current_point];
    float anti_grav = anti_gravity[current_point];

    int point_vertex_reference = point_vertex_references[current_point];
    int point_hull_index = point_hull_indices[current_point];
    int point_flag = point_flags[current_point];

    int4 bone_table = bone_tables[current_point];
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (shift > 0)
    {
        int new_point_index = current_point - shift;
        points[new_point_index] = point;
        anti_gravity[new_point_index] = anti_grav;
        point_vertex_references[new_point_index] = point_vertex_reference; 
        point_hull_indices[new_point_index] = point_hull_index; 
        point_flags[new_point_index] = point_flag; 
        bone_tables[new_point_index] = bone_table;
    }
}

__kernel void compact_bones(__global int *bone_shift,
                            __global float16 *bone_instances,
                            __global int *hull_bind_pose_indicies,
                            __global int *hull_inv_bind_pose_indicies)
{
    int current_bone = get_global_id(0);
    int shift = bone_shift[current_bone];
    float16 instance = bone_instances[current_bone];
    int bind_pose_id = hull_bind_pose_indicies[current_bone];
    int inv_bind_pose_id = hull_inv_bind_pose_indicies[current_bone];
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (shift > 0)
    {
        int new_bone_index = current_bone - shift;
        bone_instances[new_bone_index] = instance;
        hull_bind_pose_indicies[current_bone] = bind_pose_id;
        hull_inv_bind_pose_indicies[current_bone] = inv_bind_pose_id;
    }
}

__kernel void compact_armature_bones(__global int *bone_bind_shift,
                                     __global float16 *armatures_bones,
                                     __global int2 *bind_tables)
{
    int current_armature_bone = get_global_id(0);
    int shift = bone_bind_shift[current_armature_bone];
    float16 armature_bone = armatures_bones[current_armature_bone];
    int2 bind_table = bind_tables[current_armature_bone];
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (shift > 0)
    {
        int new_bone_index = current_armature_bone - shift;
        armatures_bones[new_bone_index] = armature_bone;
        bind_tables[new_bone_index] = bind_table;
    }
}