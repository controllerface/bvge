inline void upsweep(__local int *buffer, int m) 
{
    // the local ID for this thread
    int local_id = get_local_id(0);

    // b-index for this work item
    int b_index = (local_id * 2) + 1;

    // the depth of the "tree" we are traversing
    int max_depth = 1 + (int)log2((float)m);

    // traverse the tree upward and do the additions to calculate the value at each node
    for (int depth = 0; depth < max_depth; depth++) 
    {
        // all threads must hit this barrier to ensure local buffer data is in sync
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // calculate a mask? todo: why is it done this way?
        int mask = (0x1 << depth) - 1;
        
        // check if we are supposed to do work? todo: why is a mask used?
        if ((local_id & mask) == mask) 
        {
            int offset = (0x1 << depth);
            // a-index for this work item
            int a_index = b_index - offset;
            buffer[b_index] += buffer[a_index];
        }
    }
}

inline void downsweep(__local int *buffer, int m)
{
    // the local  ID for this thread
    int local_id = get_local_id(0);

    // b-index for this work item
    int b_index = (local_id * 2) + 1;
    int max_depth = (int)log2((float)m);

    // traverse the tree downward and do the additions to calculate the value at each node
    for (int depth = max_depth; depth > -1; depth--) 
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        int mask = (0x1 << depth) - 1;
        if ((local_id & mask) == mask) 
        {
            int offset = (0x1 << depth);
            // a-index for this work item
            int a_index = b_index - offset;
            int temp = buffer[a_index];
            buffer[a_index] = buffer[b_index];
            buffer[b_index] += temp;
        }
    }
}

__kernel void scan_bounds_single_block(__global float16 *bounds,
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
    }

    if (b_index < n) 
    {
        bounds[b_index].s4 = (float)buffer[b_index] / 2;
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
    }
    if (b_index < n) 
    {
        bounds[b_index].s4 = (float)buffer[local_b_index] / 2;
    }
}

__kernel void scan_int_single_block(__global int *data, __local int *buffer, int n) 
{
    // the global ID for this thread
    int global_id = get_global_id(0);

    // determine the IDs for this thread in the local data buffer
    int a_index = (global_id * 2);
    int b_index = (global_id * 2) + 1;

    // calculate the total length of the local buffer array
    int m = 2 * get_local_size(0);

    // load the gloabl value sinto the local buffer, and with extra zeroes
    // so the buffer size matches the work group
    buffer[a_index] = a_index < n ? data[a_index] : 0;
    buffer[b_index] = b_index < n ? data[b_index] : 0;

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
        data[a_index] = buffer[a_index];
    }

    if (b_index < n) 
    {
        data[b_index] = buffer[b_index];
    }
}

/*
 * data is of some arbitrary lenght [n]
 * buffer is is size [m] where m a multiple of 2 and is 2 times the number of items in the 
 * local work group, it holds _pairs_ of entries used for intermediate operations.
 * part is a buffer of size [k] where k is the total number of blocks being processed
 * n is the total number of elements in the entire global data array.
 */
__kernel void scan_int_multi_block(__global int *data, 
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
    buffer[local_a_index] = (a_index < n) ? data[a_index] : 0;
    buffer[local_b_index] = (b_index < n) ? data[b_index] : 0;

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
        data[a_index] = buffer[local_a_index];
    }
    if (b_index < n) 
    {
        data[b_index] = buffer[local_b_index];
    }
}

__kernel void complete_int_multi_block(__global int *data, 
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
    buffer[local_a_index] = (a_index < n) ? data[a_index] : 0;
    buffer[local_b_index] = (b_index < n) ? data[b_index] : 0;

    buffer[local_a_index] += part[grpid];
    buffer[local_b_index] += part[grpid];

    // copy back to global data
    if (a_index < n) 
    {
        data[a_index] = buffer[local_a_index];
    }
    if (b_index < n) 
    {
        data[b_index] = buffer[local_b_index];
    }
}