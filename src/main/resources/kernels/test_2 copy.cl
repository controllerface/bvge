__kernel void exclusiveScan(__global float* input, __global float* output, int n, __local float* temp)

{
    int id = get_global_id(0);
    int local_id = get_local_id(0);

    // Copy global memory to shared memory
    temp[local_id] = input[id];

    // Perform upsweep (reduction) phase
    for (int stride = 1; stride <= get_local_size(0) / 2; stride *= 2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        int index = (local_id + 1) * stride * 2 - 1;
        if (index < get_local_size(0))
        {
            temp[index] += temp[index - stride];
        }
    }

    // Clear the last element
    if (local_id == get_local_size(0) - 1)
    {
        temp[get_local_size(0) - 1] = 0;
    }

    // Perform downsweep phase
    for (int stride = get_local_size(0) / 2; stride > 0; stride /= 2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        int index = (local_id + 1) * stride * 2 - 1;
        if (index < get_local_size(0))
        {
            float temp_value = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += temp_value;
        }
    }

    // Copy the exclusive scan result from shared memory to global memory
    output[id] = temp[local_id];

    // Adjust the first element for exclusive scan
    if (local_id == 0 && id > 0)
    {
        output[id] = temp[local_id - 1];
    }
}
