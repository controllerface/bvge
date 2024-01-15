__kernel void scan_int4_single_block(__global int4 *data, 
                                     __local int4 *buffer, 
                                     int n) 
{
    // the global ID for this thread
    int global_id = get_global_id(0);

    // determine the IDs for this thread in the local data buffer
    int a_index = (global_id * 2);
    int b_index = (global_id * 2) + 1;

    // calculate the total length of the local buffer array
    int m = 2 * get_local_size(0);

    // load the global values into the local buffer, with extra zeroes
    // if needed, so the buffer size matches the work group size
    buffer[a_index] = a_index < n ? data[a_index] : 0;
    buffer[b_index] = b_index < n ? data[b_index] : 0;

    // perform the "up sweep" traversing from leaf to root of the tree
    upsweep_vec4(buffer, m);

    // set lane b to 0 if the count was odd, since the b index will be unused
    if (b_index == (m - 1)) 
    {
        buffer[b_index] = 0;
    }

    // perform the "down sweep" from root back down to the leaves
    downsweep_vec4(buffer, m);

    // for each index that is in range, copy the local buffer to the data buffer
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
__kernel void scan_int4_multi_block(__global int4 *data, 
                                    __local int4 *buffer, 
                                    __global int4 *part, 
                                    int n)
{
    // workgroup size
    int wx = get_local_size(0);

    // global identifiers and indicies
    int global_id = get_global_id(0);
    int a_index = (2 * global_id);
    int b_index = (2 * global_id) + 1;

    // local identifiers and indicies
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

    // a reduce on each subarray
    upsweep_vec4(buffer, m);

    // last workitem per workgroup saves last element of each subarray in [part]
    // before zeroing
    if (local_id == (wx - 1)) 
    {
        part[grpid] = buffer[local_b_index];
        buffer[local_b_index] = 0;
    }

    // a sweepdown on each subarray
    downsweep_vec4(buffer, m);

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

__kernel void complete_int4_multi_block(__global int4 *data, 
                                        __local int4 *buffer, 
                                        __global int4 *part, 
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
