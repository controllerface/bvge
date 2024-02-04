
constant float16 identity_matrix = (float16)
(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
);

inline float16 matrix_mul_affine(float16 matrixA, float16 matrixB) 
{
    float16 result;

    float m00 = matrixA.s0;
    float m01 = matrixA.s1;
    float m02 = matrixA.s2;
    float m10 = matrixA.s4;
    float m11 = matrixA.s5;
    float m12 = matrixA.s6;
    float m20 = matrixA.s8;
    float m21 = matrixA.s9;
    float m22 = matrixA.sA;

    float rm00 = matrixB.s0;
    float rm01 = matrixB.s1;
    float rm02 = matrixB.s2;
    float rm10 = matrixB.s4;
    float rm11 = matrixB.s5;
    float rm12 = matrixB.s6;
    float rm20 = matrixB.s8;
    float rm21 = matrixB.s9;
    float rm22 = matrixB.sA;
    float rm30 = matrixB.sC;
    float rm31 = matrixB.sD;
    float rm32 = matrixB.sE;

    result.s0 = fma(m00, rm00, fma(m10, rm01, m20 * rm02));
    result.s1 = fma(m01, rm00, fma(m11, rm01, m21 * rm02)); 
    result.s2 = fma(m02, rm00, fma(m12, rm01, m22 * rm02)); 
    result.s3 = matrixA.s3; 
    result.s4 = fma(m00, rm10, fma(m10, rm11, m20 * rm12)); 
    result.s5 = fma(m01, rm10, fma(m11, rm11, m21 * rm12)); 
    result.s6 = fma(m02, rm10, fma(m12, rm11, m22 * rm12)); 
    result.s7 = matrixA.s7; 
    result.s8 = fma(m00, rm20, fma(m10, rm21, m20 * rm22)); 
    result.s9 = fma(m01, rm20, fma(m11, rm21, m21 * rm22)); 
    result.sA = fma(m02, rm20, fma(m12, rm21, m22 * rm22)); 
    result.sB = matrixA.sB; 
    result.sC = fma(m00, rm30, fma(m10, rm31, fma(m20, rm32, matrixA.sC))); 
    result.sD = fma(m01, rm30, fma(m11, rm31, fma(m21, rm32, matrixA.sD))); 
    result.sE = fma(m02, rm30, fma(m12, rm31, fma(m22, rm32, matrixA.sE))); 
    result.sF = matrixA.sF; 

    return result;
}

__kernel void animate_armatures(__global float16 *armature_bones,
                                __global float16 *bone_bind_poses,
                                __global float16 *model_transforms,
                                __global int2 *bone_bind_tables,
                                __global int4 *armature_flags,
                                __global int4 *hull_tables)
{
    int current_armature = get_global_id(0);
    int4 hull_table = hull_tables[current_armature];
    int4 armature_flag = armature_flags[current_armature];
    float16 model_transform = model_transforms[armature_flag.w];

    // note that armatures with no bones simply do nothing as the bone count will be zero
    int armature_bone_count = hull_table.w - hull_table.z + 1;
    for (int i = 0; i < armature_bone_count; i++)
    {
        int current_bone_bind = hull_table.z + i;
        int2 bone_bind_table = bone_bind_tables[current_bone_bind];

        float16 parent_transform = bone_bind_table.y == -1 
            ? model_transform 
            : armature_bones[bone_bind_table.y];

        // todo: when there is animation, there will be a check to determine where the transform comes from
        float16 node_transform = bone_bind_poses[bone_bind_table.x];
        float16 global_transform = matrix_mul_affine(parent_transform, node_transform);
        armature_bones[current_bone_bind] = global_transform;
    }
}

__kernel void animate_bones(__global float16 *bones,
                            __global float16 *bone_references,
                            __global float16 *armature_bones,
                            __global int2 *bone_index_tables)
{
    int current_bone = get_global_id(0);
    int2 index_table = bone_index_tables[current_bone];
    float16 bone_reference = bone_references[index_table.x];
    float16 armature_bone = armature_bones[index_table.y];
    float16 next_position = matrix_mul_affine(armature_bone, bone_reference);
    bones[current_bone] = next_position;
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

    float16 bone1 = bone_table.x == -1 ? identity_matrix : bones[bone_table.x];
    float16 bone2 = bone_table.y == -1 ? identity_matrix : bones[bone_table.y];
    float16 bone3 = bone_table.z == -1 ? identity_matrix : bones[bone_table.z];
    float16 bone4 = bone_table.w == -1 ? identity_matrix : bones[bone_table.w];

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
