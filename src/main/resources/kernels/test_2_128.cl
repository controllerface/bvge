// this is probably not correct

__kernel void exclusiveScan(__global float* input, __global float* output, int n)
{
    // Allocate shared memory
    __local float temp[128];

    int id = get_global_id(0);
    int local_id = get_local_id(0);

    // Copy global memory to shared memory
    temp[local_id] = input[id];

    // Perform upsweep (reduction) phase
    for (int stride = 1; stride <= 64; stride *= 2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        int index = (local_id + 1) * stride * 2 - 1;
        if (index < 128)
        {
            temp[index] += temp[index - stride];
        }
    }

    // Clear the last element
    if (local_id == 127)
    {
        temp[127] = 0;
    }

    // Perform downsweep phase
    for (int stride = 64; stride > 0; stride /= 2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        int index = (local_id + 1) * stride * 2 - 1;
        if (index < 128)
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
