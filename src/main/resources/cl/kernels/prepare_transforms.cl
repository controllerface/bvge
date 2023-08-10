__kernel void prepare_transforms(__global float4 *transforms, 
                                 __global float2 *hull_rotations,
                                 __global int *indices,
                                 __global float4 *transforms_out)
{
    int gid = get_global_id(0);
    int index = indices[gid];
    
    float4 transform = transforms[index];
    float2 rotation = hull_rotations[index];

    float4 transform_out;
    transform_out.x = transform.x; 
    transform_out.y = transform.y; 
    transform_out.z = rotation.x;
    transform_out.w = transform.z; // just use x scale for now
    // todo: will need to expand this structure out for non-uniform scales and circles

    transforms_out[gid] = transform_out;
}
