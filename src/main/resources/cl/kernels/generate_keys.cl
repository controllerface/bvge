
// todo: convert to int 2, key bank and int 4, aabb index

__kernel void generate_keys(__global float16 *bounds,
                            __global int2 *bounds_bank_data,
                            __global int *key_bank,
                            volatile __global int *key_counts,
                            int x_subdivisions,
                            int key_bank_length,
                            int key_count_length)
{
    int gid = get_global_id(0);
    float16 bound = bounds[gid];
    int2 bounds_bank = bounds_bank_data[gid];

    bool inBounds = bound.s5 != 0;
    if (!inBounds)
    {
        return;
    }

    int offset = bound.s4 * 2;

    int min_x = bound.s6;
    int max_x = bound.s7;
    int min_y = bound.s8;
    int max_y = bound.s9;

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
