__kernel void prepare_bones(__global float16 *bones,
                            __global float16 *bone_references, 
                            __global int *bone_index, 
                            __global float4 *hulls,
                            __global float4 *armatures,
                            __global int2 *hull_flags,
                            __global float2 *vbo,
                            int offset)
{
    int gid = get_global_id(0);
    int bone_id = gid + offset;
    
    float4 hull = hulls[bone_id];
    float16 bone = bones[bone_id];
    int2 hull_1_flags = hull_flags[bone_id];
    float4 armature = armatures[hull_1_flags.y];

    int bone_idx = bone_index[bone_id];
    float16 bone_ref = bone_references[bone_idx];

    float4 result = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
    result = matrix_transform(bone_ref, result);
    //result = matrix_transform(bone, result);

   

    result.x *= hull.z;
    result.y *= hull.w;
    result.x += armature.x;
    result.y += armature.y;


    
    // float4 padded = (float4)(hull.x, hull.y,  0.0f, 0.0f);

    // float4 result = matrix_transform(bone, padded);
    // padded.x += hull.x;
    // padded.y += hull.y;
    
    //vbo[gid] = armature.xy;
    vbo[gid] = armature.xy;
}
