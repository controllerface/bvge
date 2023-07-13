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

__kernel void scan_key_bank(__global float16 *bounds,
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

    // load the gloabl value sinto the local buffer, and with extra zeroes
    // so the buffer size matches the work group
    buffer[a_index] = a_index < n ? bounds[a_index].s5 : 0;
    buffer[b_index] = b_index < n ? bounds[b_index].s5 : 0;

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
        bounds[a_index].s4 = buffer[a_index] / 2;
    }

    if (b_index < n) 
    {
        bounds[b_index].s4 = buffer[b_index] / 2;
    }
}