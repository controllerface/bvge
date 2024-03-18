__kernel void buffer_transfer(__global float4 *input, __global float4 *output)
{
    int current_edge = get_global_id(0);
    if (current_edge > 64415 && current_edge < 64420)
    {
        float4 in = input[current_edge];
        printf("edge [%d] in: %d %d %f %d", current_edge, (int)in.x, (int)in.y, in.z, (int)in.w);
    }
    output[current_edge] = input[current_edge];
}

__kernel void verify_buffer_transfer(__global float4 *input, __global float4 *output)
{
    int current_edge = get_global_id(0);
    if (current_edge > 64415 && current_edge < 64420)
    {
        float4 out = output[current_edge];
        printf("edge [%d] out: %d %d %f %d", current_edge, (int)out.x, (int)out.y, out.z, (int)out.w);
    }

    // if (output[current_edge] != input[current_edge])
    // {
    //     // float4 ex = input[current_edge];
    //     // float4 ac = output[current_edge];
    //     // printf("error: expected: %d %d %f %d saw: %d %d %f %d", 
    //     //     ex.x, ex.y, ex.z, ex.w, 
    //     //     ac.x, ac.y, ac.z, ac.w);
    // }
}