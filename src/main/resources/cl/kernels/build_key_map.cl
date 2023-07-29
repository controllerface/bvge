
// todo: convert to int 2, key bank and int 4, aabb index

__kernel void build_key_map(__global int4 *bounds_index_data,
                            __global int2 *bounds_bank_data,
                            __global int *key_map,
                            __global int *key_offsets,
                            __global int *key_counts,
                            int x_subdivisions,
                            int key_count_length)
{
    int gid = get_global_id(0);
    int4 bounds_index = bounds_index_data[gid];
    int2 bounds_bank = bounds_bank_data[gid];

    bool inBounds = bounds_bank.y != 0;
    if (!inBounds)
    {
        return;
    }

    int min_x = bounds_index.x;
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
            key_map[offset + i] = gid;
        }
    }
}
