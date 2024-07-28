
/**
Locates objects that are in bounds of the spatial partition
 */
__kernel void locate_in_bounds(__global int2 *bounds_bank_data,
                               __global int *in_bounds,
                               __global int *counter,
                               int max_bound)
{
    int current_hull = get_global_id(0);
    if (current_hull >= max_bound) return;
    int2 bounds_bank = bounds_bank_data[current_hull];
    bool is_in_bounds = bounds_bank.y > 0;
    if (is_in_bounds)
    {
        int i = atomic_inc(&counter[0]);
        in_bounds[i] = current_hull;
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
                               int key_count_length, 
                               int max_index)
{
    int current_candidate = get_global_id(0);
    if (current_candidate >= max_index) return;
    int index = in_bounds[current_candidate];
    int2 bounds_bank = bounds_bank_data[index];
    int spatial_index = bounds_bank.x * 2;
    int spatial_length = bounds_bank.y;
    int end = spatial_index + spatial_length;

    int size = 0;
    __attribute__((opencl_unroll_hint(8)))
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
    candidates[current_candidate].x = index;
    candidates[current_candidate].y = size;
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
                                  __global int2 *final_candidates,
                                  int max_index)
{
    int current_candidate = get_global_id(0);
    if (current_candidate >= max_index) return;
    int index = input_candidates[current_candidate].x;
    int size = used[current_candidate];
    int offset = match_offsets[current_candidate];
    __attribute__((opencl_unroll_hint(8)))
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