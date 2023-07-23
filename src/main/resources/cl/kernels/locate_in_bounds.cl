__kernel void locate_in_bounds(__global float16 *bounds,
                               __global int *in_bounds,
                               __global int *counter)
{
    int gid = get_global_id(0);
    float16 bound = bounds[gid];
    bool is_in_bounds = bound.s5 > 0;
    if (is_in_bounds)
    {
        int i = atomic_inc(&counter[0]);
        in_bounds[i] = gid;
    }    
}

__kernel void count_candidates(__global float16 *bounds,
                               __global int *in_bounds,
                               __global int *key_bank,
                               __global int *key_counts,
                               __global int2 *candidates,
                               int x_subdivisions,
                               int key_count_length)
{
    int gid = get_global_id(0);
    int index = in_bounds[gid];
    float16 bound = bounds[index];

    int spatial_index = (int)bound.s4 * 2;
    int spatial_length = (int)bound.s5;
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
        int2 pair = (int2)(index, next);
        int j = atomic_inc(&counter[0]);
        final_candidates[j] = pair;
    }
}
