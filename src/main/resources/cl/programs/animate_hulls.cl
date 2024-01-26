
constant float16 identityMatrix = (float16)
(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
);


__kernel void animate_armatures()
{
    // todo: stub out

}

__kernel void animate_hulls()
{
    // get hull bone table
    // get bone instances
    // get bone index tables
    // get mesh-space inverse bind matrix
    // get armature aligned current animation

}

__kernel void animate_points(__global float4 *points,
                            __global float4 *hulls,
                            __global int4 *hull_flags,
                            __global int4 *vertex_tables,
                            __global int4 *bone_tables,
                            __global float4 *vertex_weights,
                            __global float4 *armatures,
                            __global float2 *vertex_references,
                            __global float16 *bones)
{

    int gid = get_global_id(0);
    float4 point = points[gid];
    int4 vertex_table = vertex_tables[gid];
    int4 hull_flag = hull_flags[vertex_table.y];
    bool no_bones = (hull_flag.x & NO_BONES) !=0;
    if (no_bones) return;

    int4 bone_table = bone_tables[gid];

    float2 reference_vertex = vertex_references[vertex_table.x];
    float4 reference_weights = vertex_weights[vertex_table.x];

    float16 bone1 = bone_table.x == -1 ? identityMatrix : bones[bone_table.x];
    float16 bone2 = bone_table.y == -1 ? identityMatrix : bones[bone_table.y];
    float16 bone3 = bone_table.z == -1 ? identityMatrix : bones[bone_table.z];
    float16 bone4 = bone_table.w == -1 ? identityMatrix : bones[bone_table.w];

    float16 test_bone = bone1 * reference_weights.x;
    test_bone += bone2 * reference_weights.y;
    test_bone += bone3 * reference_weights.z;
    test_bone += bone4 * reference_weights.w;
    
    float4 hull = hulls[vertex_table.y];
    float4 armature = armatures[hull_flag.y]; 

    float4 padded = (float4)(reference_vertex.x, reference_vertex.y, 0.0f, 1.0f);
    float4 after_bone = matrix_transform(test_bone, padded);
    float2 un_padded = after_bone.xy;
    
    // this is effectively a model transform with just scale and position
    un_padded.x *= hull.z;
    un_padded.y *= hull.w;
    un_padded += armature.xy;
    point.x = un_padded.x;
    point.y = un_padded.y;
    points[gid] = point;
}
