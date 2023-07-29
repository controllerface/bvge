/**
Performs axis-aligned bounding box collision detection as part of a broad phase collision step.
 */
__kernel void aabb_collide(__global float4 *bounds,
                           __global int2 *bounds_bank_data,
                           __global int2 *candidates,
                           __global int *match_offsets,
                           __global int *key_map,
                           __global int *key_bank,
                           __global int *key_counts,
                           __global int *key_offsets,
                           __global int *matches,
                           __global int *used,
                           volatile __global int *counter,
                           int x_subdivisions,
                           int key_count_length)
{
    int gid = get_global_id(0);
    int index = candidates[gid].x;
    int size = candidates[gid].y;
    int match_offset = match_offsets[gid];

    float4 bound = bounds[index];
    int2 bounds_bank = bounds_bank_data[index];

    int spatial_index = bounds_bank.x * 2;
    int spatial_length = bounds_bank.y;

    int end = spatial_index + spatial_length;

    int current_offset = match_offset;
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

        int key_offset = key_offsets[key_index];

        // loop through all the candidates at this key
        for (int map_index = key_offset; map_index < key_offset + count; map_index++)
        {
            int next = key_map[map_index]; 

            // no mirror or self-matches
            if (index >= next)
            {
                continue;
            }

            // broad phase collision check
            float4 candidate = bounds[next];
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
                for (int match_index = match_offset; match_index < current_offset; match_index++)
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
            matches[current_offset++] = next;
            slots_used++;
        }
    }

    used[gid] = slots_used;
    atomic_add(&counter[0], slots_used);
}