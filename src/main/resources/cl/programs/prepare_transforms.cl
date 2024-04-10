__kernel void prepare_transforms(__global float2 *hull_positions, 
                                 __global float2 *hull_scales, 
                                 __global float2 *hull_rotations,
                                 __global int *indices,
                                 __global float4 *transforms_out,
                                 int offset)
{
    int gid = get_global_id(0);
    int offset_gid = gid + offset;
    int index = indices[offset_gid];
    
    float2 position = hull_positions[index];
    float2 scale    = hull_scales[index];
    float2 rotation = hull_rotations[index];

    float4 transform_out;
    transform_out.x = position.x; 
    transform_out.y = position.y; 
    transform_out.z = rotation.x;
    transform_out.w = scale.x; // note: uniform scale only

    transforms_out[gid] = transform_out;
}
