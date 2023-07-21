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