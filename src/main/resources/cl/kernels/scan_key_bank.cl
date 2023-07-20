__kernel void scan_bounds_single_block(__global float16 *bounds,
                                       __global int *sz,
                                       __local int *buffer,
                                       int n)
{
    // the global ID for this thread
    int global_id = get_global_id(0);

    // determine the IDs for this thread in the local data buffer
    int a_index = (global_id * 2);
    int b_index = (global_id * 2) + 1;

    // calculate the total length of the local buffer array
    int m = 2 * get_local_size(0);

    // load the gloabl values into the local buffer, and with extra zeroes
    // so the buffer size matches the work group
    buffer[a_index] = a_index < n ? (int)bounds[a_index].s5 : 0;
    buffer[b_index] = b_index < n ? (int)bounds[b_index].s5 : 0;

    // perform the "up sweep" traversing from leaf to root of the tree
    upsweep(buffer, m);

    // set lane 1 to 0? todo: determine why
    if (b_index == (m - 1)) 
    {
        buffer[b_index] = 0;
    }

    downsweep(buffer, m);

    if (a_index < n) 
    {
        bounds[a_index].s4 = (float)buffer[a_index] / 2;
        if (a_index == n - 1)
        {
            sz[0] = (bounds[a_index].s4 + bounds[a_index].s5) * 2;
        }
    }

    if (b_index < n) 
    {
        bounds[b_index].s4 = (float)buffer[b_index] / 2;
        if (b_index == n - 1)
        {
            sz[0] = (bounds[b_index].s4 + bounds[b_index].s5) * 2;
        }
    }
}

__kernel void scan_bounds_multi_block(__global float16 *bounds, 
                                  __local int *buffer, 
                                  __global int *part, 
                                  int n)
{
    // workgroup size
    int wx = get_local_size(0);

    // global identifiers and indexes
    int global_id = get_global_id(0);
    int a_index = (2 * global_id);
    int b_index = (2 * global_id) + 1;

    // local identifiers and indexes
    int local_id = get_local_id(0);
    int local_a_index = (2 * local_id);
    int local_b_index = (2 * local_id) + 1;
    int grpid = get_group_id(0);

    // list lengths
    int m = wx * 2;
    int k = get_num_groups(0);

    // copy into local data padding elements >= n with 0
    buffer[local_a_index] = (a_index < n) ? (int)bounds[a_index].s5 : 0;
    buffer[local_b_index] = (b_index < n) ? (int)bounds[b_index].s5 : 0;

    // ON EACH SUBARRAY
    // a reduce on each subarray
    upsweep(buffer, m);

    // last workitem per workgroup saves last element of each subarray in [part]
    // before zeroing
    if (local_id == (wx - 1)) 
    {
        part[grpid] = buffer[local_b_index];
        buffer[local_b_index] = 0;
    }

    // a sweepdown on each subarray
    downsweep(buffer, m);

    // copy back to global data
    if (a_index < n) 
    {
        bounds[a_index].s4 = (float)buffer[local_a_index];
    }
    if (b_index < n) 
    {
        bounds[b_index].s4 = (float)buffer[local_b_index];
    }
}

__kernel void complete_bounds_multi_block(__global float16 *bounds, 
                                          __global int *sz,
                                          __local int *buffer, 
                                          __global int *part, 
                                          int n)
{
    // global identifiers and indexes
    int global_id = get_global_id(0);
    int a_index = (2 * global_id);
    int b_index = (2 * global_id) + 1;

    // local identifiers and indexes
    int local_id = get_local_id(0);
    int local_a_index = (2 * local_id);
    int local_b_index = (2 * local_id) + 1;
    int grpid = get_group_id(0);

    // copy into local data padding elements >= n with identity
    buffer[local_a_index] = (a_index < n) ? (int)bounds[a_index].s4 : 0;
    buffer[local_b_index] = (b_index < n) ? (int)bounds[b_index].s4 : 0;
    buffer[local_a_index] += part[grpid];
    buffer[local_b_index] += part[grpid];

    // copy back to global data
    if (a_index < n) 
    {
        bounds[a_index].s4 = (float)buffer[local_a_index] / 2;
        if (a_index == n - 1)
        {
            sz[0] = (bounds[a_index].s4 + bounds[a_index].s5) * 2;
        }
    }
    if (b_index < n) 
    {
        bounds[b_index].s4 = (float)buffer[local_b_index] / 2;
        if (b_index == n - 1)
        {
            sz[0] = (bounds[b_index].s4 + bounds[b_index].s5) * 2;
        }
    }
}