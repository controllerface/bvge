__kernel void addOffset(__global float* input, __global float* output, int n, float offset)
{
    size_t globalId = get_global_id(0);
    
    if (globalId < n)
    {
        output[globalId] = input[globalId] + offset;
    }
}