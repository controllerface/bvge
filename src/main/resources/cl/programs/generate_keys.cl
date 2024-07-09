
/**
Generates the spatial index keys for each object and stores them in the key bank.
 */
__kernel void generate_keys(__global int4 *bounds_index_data,
                            __global int2 *bounds_bank_data,
                            __global int *key_bank,
                            __global int *key_counts,
                            int x_subdivisions,
                            int key_bank_length,
                            int key_count_length, 
                            int max_hull)
{
    int current_hull = get_global_id(0);
    if (current_hull >= max_hull) return;
    int4 bounds_index = bounds_index_data[current_hull];
    int2 bounds_bank = bounds_bank_data[current_hull];

    int offset = bounds_bank.x * 2;

    int min_x = bounds_bank.y == 0 ? INT_MAX : bounds_index.x; // ensure out of bounds hulls never enter the loop
    int max_x = bounds_index.y;
    int min_y = bounds_index.z;
    int max_y = bounds_index.w;

    int current_index = offset;
    for (int current_x = min_x; current_x <= max_x; current_x++)
    {
        for (int current_y = min_y; current_y <= max_y; current_y++)
        {
            int key_index = calculate_key_index(x_subdivisions, current_x, current_y);
            if (key_index < 0 || current_index < 0
                || key_index >= key_count_length
                || current_index >= key_bank_length)
            {
                continue;
            }

            key_bank[current_index++] = current_x;
            key_bank[current_index++] = current_y;
            atomic_inc(&key_counts[key_index]);
        }
    }
}
