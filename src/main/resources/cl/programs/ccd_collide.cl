
inline int cmp(float a, float b) 
{
	return (a > b ? 1 : 0) - (a < b ? 1 : 0);
}

inline int orientation(float x0, float y0, float x1, float y1, float x2, float y2) 
{
	return cmp((x2 - x1) * (y1 - y0), (x1 - x0) * (y2 - y1));
}

inline bool linesIntersect(float4 line1, float4 line2) 
{
	return orientation(line1.x, line1.y, line1.z, line1.w, line2.x, line2.y) != orientation(line1.x, line1.y, line1.z, line1.w, line2.z, line2.w) 
        && orientation(line2.x, line2.y, line2.z, line2.w, line1.x, line1.y) != orientation(line2.x, line2.y, line2.z, line2.w, line1.z, line1.w);
}



/**
Performs axis-aligned bounding box collision detection as part of a broad phase collision step.
 */
__kernel void ccd_collide(__global int2 *edges,
                          __global float4 *points,
                          __global float4 *bounds,
                          __global int2 *bounds_bank_data,
                          __global int *hull_entity_ids,
                          __global int *hull_flags,
                          __global int *point_hull_indices,
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
                          int key_count_length,
                          int max_index)
{
    int gid = get_global_id(0);
    if (gid >= max_index) return;
    int index = candidates[gid].x;
    int size = candidates[gid].y;
    int match_offset = match_offsets[gid];

    float4 bound = bounds[index];
    int2 bounds_bank = bounds_bank_data[index];

    int2 edge = edges[index];
    int hull_id = point_hull_indices[edge.x];

    int flags = hull_flags[hull_id];
    int entity_id = hull_entity_ids[hull_id];

    int spatial_index = bounds_bank.x * 2;
    int spatial_length = bounds_bank.y;

    int end = spatial_index + spatial_length;

    int current_offset = match_offset;
    int slots_used = 0;

    bool is_static = (flags & IS_STATIC) !=0;
    bool non_colliding = (flags & NON_COLLIDING) !=0;

    if (non_colliding) return;

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
            int2 c_edge = edges[next];
            int c_hull_id = point_hull_indices[c_edge.x];

            // no mirror or self-matches
            if (index >= next)
            {
                continue;
            }

            // no collisions between hulls that are part of the same amrature
            int candiate_entity_id = hull_entity_ids[c_hull_id];
            if (candiate_entity_id == entity_id)
            {
                continue;
            }

            int candiate_flags = hull_flags[c_hull_id];
            bool is_static_c = (candiate_flags & IS_STATIC) !=0;

            // no static/static collision permitted
            if (is_static && is_static_c)
            {
                continue;
            }

            if (!is_static && !is_static_c)
            {
                continue;
            }

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

            //printf("edge 1: %d edge 2: %d", index, next);
            int static_edge_id = is_static 
                ? index 
                : next;
            int non_static_edge_id = is_static_c 
                ? index 
                : next;

            int2 static_edge = edges[static_edge_id];
            int2 non_static_edge = edges[non_static_edge_id];
            
            float4 static_line = (float4)(points[static_edge.x].xy, points[static_edge.y].xy);
            float4 dyn_line_1 = points[non_static_edge.x];
            float4 dyn_line_2 = points[non_static_edge.y];

            if (linesIntersect(static_line, dyn_line_1))
            {
                printf("static edge: %d dyn edge 1: %d", static_edge_id, non_static_edge_id);
            }
            else if (linesIntersect(static_line, dyn_line_2))
            {
                printf("static edge: %d dyn edge 2: %d", static_edge_id, non_static_edge_id);
            }
        }
    }

    used[gid] = slots_used;
    atomic_add(&counter[0], slots_used);
}
