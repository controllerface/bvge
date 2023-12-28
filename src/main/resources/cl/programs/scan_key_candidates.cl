__kernel void scan_candidates_single_block_out(__global int2 *input, 
                                               __global int *output,
                                               __global int *sz,
                                               __local int *buffer, 
                                               int n) 
{
    int global_id = get_global_id(0);

    int a_index = (global_id * 2);
    int b_index = (global_id * 2) + 1;

    int m = 2 * get_local_size(0);

    buffer[a_index] = a_index < n ? input[a_index].y : 0;
    buffer[b_index] = b_index < n ? input[b_index].y : 0;

    upsweep(buffer, m);

    if (b_index == (m - 1)) 
    {
        buffer[b_index] = 0;
    }

    downsweep(buffer, m);

    if (a_index < n) 
    {
        output[a_index] = buffer[a_index];
        if (a_index == n - 1)
        {
            sz[0] = (output[a_index] + input[a_index].y);
        }
    }

    if (b_index < n) 
    {
        output[b_index] = buffer[b_index];
        if (b_index == n - 1)
        {
            sz[0] = (output[b_index] + input[b_index].y);
        }
    }
}

__kernel void scan_candidates_multi_block_out(__global int2 *input, 
                                              __global int *output,
                                              __local int *buffer, 
                                              __global int *part, 
                                              int n)
{
    int wx = get_local_size(0);

    int global_id = get_global_id(0);
    int a_index = (2 * global_id);
    int b_index = (2 * global_id) + 1;

    int local_id = get_local_id(0);
    int local_a_index = (2 * local_id);
    int local_b_index = (2 * local_id) + 1;
    int grpid = get_group_id(0);

    int m = wx * 2;
    int k = get_num_groups(0);

    buffer[local_a_index] = (a_index < n) ? input[a_index].y : 0;
    buffer[local_b_index] = (b_index < n) ? input[b_index].y : 0;

    upsweep(buffer, m);

    if (local_id == (wx - 1)) 
    {
        part[grpid] = buffer[local_b_index];
        buffer[local_b_index] = 0;
    }

    downsweep(buffer, m);

    if (a_index < n) 
    {
        output[a_index] = buffer[local_a_index];
    }
    if (b_index < n) 
    {
        output[b_index] = buffer[local_b_index];
    }
}

__kernel void complete_candidates_multi_block_out(__global int2 *input,
                                                  __global int *output,
                                                  __global int *sz,
                                                  __local int *buffer, 
                                                  __global int *part, 
                                                  int n)
{
    int global_id = get_global_id(0);
    int a_index = (2 * global_id);
    int b_index = (2 * global_id) + 1;

    int local_id = get_local_id(0);
    int local_a_index = (2 * local_id);
    int local_b_index = (2 * local_id) + 1;
    int grpid = get_group_id(0);

    buffer[local_a_index] = (a_index < n) ? output[a_index] : 0;
    buffer[local_b_index] = (b_index < n) ? output[b_index] : 0;

    buffer[local_a_index] += part[grpid];
    buffer[local_b_index] += part[grpid];

    if (a_index < n) 
    {
        output[a_index] = buffer[local_a_index];
        if (a_index == n - 1)
        {
            sz[0] = (output[a_index] + input[a_index].y);
        }
    }
    if (b_index < n) 
    {
        output[b_index] = buffer[local_b_index];
        if (b_index == n - 1)
        {
            sz[0] = (output[b_index] + input[b_index].y);
        }
    }
}
