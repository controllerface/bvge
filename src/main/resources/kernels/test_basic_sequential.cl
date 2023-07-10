__kernel void scan(__global const float* input, __global float* output, const int n) 
{
    int globalId = get_global_id(0);
    if (globalId > 0) 
    {
        float sum = 0;
        for (int i = 0; i < globalId; i++) 
        {
            sum += input[i];
        }
        output[globalId] = sum;
    } 
    else 
    {
        output[globalId] = 0;
    }
}