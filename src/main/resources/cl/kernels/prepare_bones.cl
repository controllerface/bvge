/**
Prepares a vbo for rendering of edges as lines. The vbo will contain only a subset of 
the vertices that make up an edge, the star of the subset is defined by the offset value.
 */
__kernel void prepare_bones(__global float16 *bones,
                            __global float16 *bone_references, 
                            __global int *bone_index, 
                            __global float4 *hulls,
                            __global float2 *vbo,
                            int offset)
{
    int gid = get_global_id(0);
    int bone_id = gid + offset;
    
    float4 hull = hulls[bone_id];
    float16 bone = bones[bone_id];
    int bone_idx = bone_index[bone_id];
    float16 bone_ref = bone_references[bone_idx];
    
    float4 padded = (float4)(0, 0, 0.0f, 0.0f);

    float4 result = matrix_transform(bone, padded);
    padded.x += hull.x;
    padded.y += hull.y;
    
    vbo[gid] = hull.xy;
}
