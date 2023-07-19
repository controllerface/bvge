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

__kernel void compute_matches(__global float16 *bounds,
                              __global int2 *candidates,
                              __global int *match_offsets,
                              __global int *key_map,
                              __global int *key_bank,
                              __global int *key_counts,
                              __global int *key_offsets,
                              __global int *matches,
                              __global int *used,
                              __global int *counter,
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
    for (int bank_index = spatial_index; bank_index < end; bank_index++)
    {
        int x = key_bank[bank_index];
        int y = key_bank[bank_index + 1];
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
        for (int map_index = offset; map_index < offset + count; map_index++)
        {
            int next = key_map[map_index]; 

            // no mirror or self-matches
            if (index >= next)
            {
                continue;
            }

            // broad phase collision check
            float16 candidate = bounds[next];
            bool near = do_bounds_intersect(bound, candidate);

            // bodies are not near each other
            if (!near)
            {
                continue;
            }

            // we need to be "set-like" so any candidate we already matched 
            // with needs to be dropped
            if (slots_used > 0)
            {
                bool dupe = false;
                for (int match_index = offset; match_index < currentOffset; match_index++)
                {
                    if (matches[match_index] == next)
                    {
                        dupe = true;
                        break;
                    }
                }
                if (dupe)
                {
                    continue;
                }
            }

            // broad phase collision detected
            matches[currentOffset++] = next;
            slots_used++;
        }
    }

    used[gid] = slots_used;
    atomic_add(&counter[0], slots_used);
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
        //printf("debug pair: t: %d c: %d", index, next);
        int2 pair = (int2)(index, next);
        int j = atomic_inc(&counter[0]);
        final_candidates[j] = pair;
    }
}
