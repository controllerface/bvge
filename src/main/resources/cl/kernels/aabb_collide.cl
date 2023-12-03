/**
Performs axis-aligned bounding box collision detection as part of a broad phase collision step.
 */
__kernel void aabb_collide(__global float4 *bounds,
                           __global int2 *bounds_bank_data,
                           __global int2 *hull_flags,
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
    int2 flags = hull_flags[index];

    int spatial_index = bounds_bank.x * 2;
    int spatial_length = bounds_bank.y;

    int end = spatial_index + spatial_length;

    int current_offset = match_offset;
    int slots_used = 0;

    bool no_bones = (flags.x & NO_BONES) !=0;
    bool is_static = (flags.x & IS_STATIC) !=0;

    // loop through all the keys for this hull
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

            // todo: add layer check
            // todo: add non-interact flag check

            // no mirror or self-matches
            if (index >= next)
            {
                continue;
            }

            // no collisions between hulls that are part of the same amrature
            int2 candiate_flags = hull_flags[next];
            if (candiate_flags.y == flags.y)
            {
                continue;
            }


            bool no_bones_c = (candiate_flags.x & NO_BONES) !=0;
            bool is_static_c = (candiate_flags.x & IS_STATIC) !=0;

            // no static/static collision permitted
            if (is_static && is_static_c)
            {
                continue;
            }

            // if (no_bones != no_bones_c)
            // {
            //     if (!is_static_c && !is_static_c)
            //     {
            //         continue;
            //     }
            // }

            // broad phase collision check
            float4 candidate = bounds[next];
            bool near = do_bounds_intersect(bound, candidate);

            // hulls are not near each other
            if (!near)
            {
                continue;
            }

            // we need to be "set-like" so any candidate we already matched 
            // with need to be dropped. This does mean that this loop size grows
            // with the number of matches, 
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