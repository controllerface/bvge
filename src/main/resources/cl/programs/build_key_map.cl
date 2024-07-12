
/**
Generates the entries within spatial key map for each hull's bounding box.
 */
__kernel void build_key_map(__global int4 *hull_aabb_index,
                            __global int2 *hull_aabb_key_table,
                            __global int *key_map,
                            __global int *key_offsets,
                            __global int *key_counts,
                            int x_subdivisions,
                            int key_count_length,
                            int max_hull)
{
    int current_hull = get_global_id(0);
    if (current_hull >= max_hull) return;
    int4 bounds_index = hull_aabb_index[current_hull];
    int2 bounds_bank  = hull_aabb_key_table[current_hull];

    int min_x = bounds_bank.y == 0 ? INT_MAX : bounds_index.x;
    int max_x = bounds_index.y;
    int min_y = bounds_index.z;
    int max_y = bounds_index.w;

    for (int current_x = min_x; current_x <= max_x; current_x++)
    {
        for (int current_y = min_y; current_y <= max_y; current_y++)
        {
            int key_index = calculate_key_index(x_subdivisions, current_x, current_y);
            if (key_index < 0 || key_index >= key_count_length)
            {
                continue;
            }
            int offset = key_offsets[key_index];
            int i = atomic_inc(&key_counts[key_index]);
            key_map[offset + i] = current_hull;
        }
    }
}
