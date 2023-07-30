__kernel void prepare_transforms(__global float4 *transforms, 
                                 __global float2 *body_rotations,
                                 __global int *indices,
                                 __global float4 *transforms_out)
{
    int gid = get_global_id(0);
    int index = indices[gid];
    
    float4 transform = transforms[index];
    float2 rotation = body_rotations[index];

    float4 transform_out;
    transform_out.x = transform.x; 
    transform_out.y = transform.y; 
    transform_out.z = rotation.x; 
    transforms_out[gid] = transform_out;
    //printf("debug: %f %f %f", transform.x, transform.y, rotation.x);
    //printf("debug: %d %f %f\n", index, transform.x, transform.y);
}