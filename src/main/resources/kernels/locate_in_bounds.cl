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


// WORK IN PROGRESS vvvv
__kernel void compute_matches(__global float16 *bounds,
                              __global int2 *candidates,
                              __global int *match_offsets,
                              __global int *key_map,
                              __global int *key_bank,
                              __global int *key_counts,
                              __global int *key_offsets,
                              __global int *matches,
                              __global int *used,
                              int x_subdivisions,
                              int key_count_length)
{
    int gid = get_global_id(0);
    int index = candidates[gid].x;
    int size = candidates[gid].y;
    int offset = match_offsets[gid];

    float16 bound = bounds[index];

    int spatial_index = (int)bound.s4 * 2;
    int spatial_length = (int)bound.s5;
    int end = spatial_index + spatial_length;

    int currentOffset = offset;
    int slots_used = 0;
    // loop through all the keys for this body
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
        if (count == 0)
        {
            continue;
        }

        int offset = key_offsets[key_index];

        // loop through all the candidates at this key
        for (int j = offset; j < offset + count; j++)
        {
            int next = key_map[j]; 

            // no duplicates or self-matches
            if (next >= index)
            {
                matches[currentOffset++] = -1;
                continue;
            }

            // broad phase collision check
            float16 candidate = bounds[next];
            bool near = do_bounds_intersect(bound, candidate);

            // bodies are not near each other
            if (!near)
            {
                matches[currentOffset++] = -1;
                continue;
            }

            // todo: matches has to be treated like a set, so if we've matched once already, we need to
            // consider that match as already having been captured. Otherwise, we coudl have multiple
            // collision reactions for the same 

            // broad phase collision detected
            matches[currentOffset++] = next;
            slots_used++;
        }
    }

    used[gid] = slots_used;
}
