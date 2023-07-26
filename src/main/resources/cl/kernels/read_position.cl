__kernel void read_position(__global float16 *bodies,
                            __global float *output,
                            int target)
{
    float16 body = bodies[target];
    output[0] = body.s0;
    output[1] = body.s1;
}