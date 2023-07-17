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
                               int x_subdivisions)
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
        int count = key_counts[key_index];
        size += count;
        if (count > 0)
        {
            size -= 1;
        }
    }
    candidates[gid].x = index;
    candidates[gid].y = size;
}
