
/**
Locates objects that are in bounds of the spatial partition
 */
__kernel void locate_in_bounds(__global int2 *bounds_bank_data,
                               __global int *in_bounds,
                               __global int *counter)
{
    int gid = get_global_id(0);
    int2 bounds_bank = bounds_bank_data[gid];
    bool is_in_bounds = bounds_bank.y > 0;
    if (is_in_bounds)
    {
        int i = atomic_inc(&counter[0]);
        in_bounds[i] = gid;
    }    
}

/**
Counts the number of potential matches each hull could have.
 */
__kernel void count_candidates(__global int2 *bounds_bank_data,
                               __global int *in_bounds,
                               __global int *key_bank,
                               __global int *key_counts,
                               __global int2 *candidates,
                               int x_subdivisions,
                               int key_count_length)
{
    int gid = get_global_id(0);
    int index = in_bounds[gid];
    int2 bounds_bank = bounds_bank_data[index];
    int spatial_index = bounds_bank.x * 2;
    int spatial_length = bounds_bank.y;
    int end = spatial_index + spatial_length;

    int size = 0;
    for (int i = spatial_index; i < end; i++)
    {
        int x = key_bank[i];
        int y = key_bank[i + 1];
        int key_index = calculate_key_index(x_subdivisions, x, y);
        if (key_index < 0 || key_index >= key_count_length)
        {
            continue;
        }
        int count = key_counts[key_index];
        size += count;
    }
    candidates[gid].x = index;
    candidates[gid].y = size;
}

/**
After candidates have been counted and the buffers created, this function forwards
all the candidates into the final buffer for processing.
 */
__kernel void finalize_candidates(__global int2 *input_candidates,
                                  __global int *match_offsets,
                                  __global int *matches,
                                  __global int *used,
                                  __global int *counter,
                                  __global int2 *final_candidates)
{
    int gid = get_global_id(0);
    int index = input_candidates[gid].x;
    int size = used[gid];
    int offset = match_offsets[gid];
    for (int i = offset; i < (offset + size); i++)
    {
        int next = matches[i];
        int a  = next < index ? index : next;
        int b  = next < index ? next : index;
        int2 pair = (int2)(a, b);
        int j = atomic_inc(&counter[0]);
        final_candidates[j] = pair;
    }
}


__kernel void sat_sort_count(__global int2 *candidates,
                             __global int *hull_flags,
                             __global int *counter)
{
    int gid = get_global_id(0);
    
    int2 current_pair = candidates[gid];
    int b1_id = current_pair.x;
    int b2_id = current_pair.y;
    int hull_1_flags = hull_flags[b1_id];
    int hull_2_flags = hull_flags[b2_id];
    bool b1s = (hull_1_flags & IS_STATIC) !=0;
    bool b2s = (hull_2_flags & IS_STATIC) !=0;
    
    bool b1_is_circle = (hull_1_flags & IS_CIRCLE) !=0;
    bool b2_is_circle = (hull_2_flags & IS_CIRCLE) !=0;

    bool b1_is_block = (hull_1_flags & IS_BLOCK) !=0;
    bool b2_is_block = (hull_2_flags & IS_BLOCK) !=0;

    bool b1_is_polygon = (hull_1_flags & IS_POLYGON) !=0;
    bool b2_is_polygon = (hull_2_flags & IS_POLYGON) !=0;

    // todo: it will probably be more performant to have separate kernels for each collision type. There should
    //  be a preliminary kernel that sorts the candidate pairs so they can be run on the right kernel
    if (b1_is_polygon && b2_is_polygon) 
    {
        atomic_inc(&counter[0]);
    }
    else if (b1_is_circle && b2_is_circle) 
    {
        atomic_inc(&counter[1]);
    }
    else if (b1_is_block && b2_is_block) 
    {
        atomic_inc(&counter[2]);
    }
    else 
    {
        atomic_inc(&counter[3]);
    }
}

__kernel void sat_sort_type(__global int2 *candidates,
                            __global int *hull_flags,
                            __global int2 *sat_candidates_p,
                            __global int2 *sat_candidates_c,
                            __global int2 *sat_candidates_b,
                            __global int2 *sat_candidates_pc,
                            __global int *counter)
{
    int gid = get_global_id(0);
    
    int2 current_pair = candidates[gid];
    int b1_id = current_pair.x;
    int b2_id = current_pair.y;
    int hull_1_flags = hull_flags[b1_id];
    int hull_2_flags = hull_flags[b2_id];
    bool b1s = (hull_1_flags & IS_STATIC) !=0;
    bool b2s = (hull_2_flags & IS_STATIC) !=0;
    
    bool b1_is_circle = (hull_1_flags & IS_CIRCLE) !=0;
    bool b2_is_circle = (hull_2_flags & IS_CIRCLE) !=0;

    bool b1_is_block = (hull_1_flags & IS_BLOCK) !=0;
    bool b2_is_block = (hull_2_flags & IS_BLOCK) !=0;

    bool b1_is_polygon = (hull_1_flags & IS_POLYGON) !=0;
    bool b2_is_polygon = (hull_2_flags & IS_POLYGON) !=0;

    // todo: it will probably be more performant to have separate kernels for each collision type. There should
    //  be a preliminary kernel that sorts the candidate pairs so they can be run on the right kernel
    if (b1_is_polygon && b2_is_polygon) 
    {
        int id = atomic_inc(&counter[0]);
        sat_candidates_p[id] = current_pair;
    }
    else if (b1_is_circle && b2_is_circle) 
    {
        int id = atomic_inc(&counter[1]);
        sat_candidates_c[id] = current_pair;
    }
    else if (b1_is_block && b2_is_block) 
    {
        int id = atomic_inc(&counter[2]);
        sat_candidates_b[id] = current_pair;
    }
    else 
    {
        int id = atomic_inc(&counter[3]);
        sat_candidates_pc[id] = current_pair;
    }
}