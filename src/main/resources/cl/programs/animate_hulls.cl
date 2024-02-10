constant float16 identity_matrix = (float16)
(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
);

inline float16 get_node_transform(__global float16 *bone_bind_poses,
                                  __global int2 *bone_channel_tables,
                                  __global int2 *bone_pos_channel_tables,
                                  __global int2 *bone_rot_channel_tables,
                                  __global int2 *bone_scl_channel_tables,
                                  __global int *animation_timing_indices,
                                  __global double2 *animation_timings,
                                  float delta_time,
                                  int animtation_index,
                                  int bone_id)
{
    // todo: when there is animation, there will be a check to determine where the transform comes from
    return bone_bind_poses[bone_id];
}

__kernel void animate_armatures(__global float16 *armature_bones,
                                __global float16 *bone_bind_poses,
                                __global float16 *model_transforms,
                                __global int2 *bone_bind_tables,
                                __global int2 *bone_channel_tables,
                                __global int2 *bone_pos_channel_tables,
                                __global int2 *bone_rot_channel_tables,
                                __global int2 *bone_scl_channel_tables,
                                __global int4 *armature_flags,
                                __global int4 *hull_tables,
                                __global int *animation_timing_indices,
                                __global double2 *animation_timings,
                                float delta_time)
{
    int current_armature = get_global_id(0);
    int4 hull_table = hull_tables[current_armature];
    int4 armature_flag = armature_flags[current_armature];
    float16 model_transform = model_transforms[armature_flag.w];

    int current_animation = 0; // todo: get from armature,
                               //  maybe use -1 for unanimated models that have bones?

    // note that armatures with no bones simply do nothing as the bone count will be zero
    int armature_bone_count = hull_table.w - hull_table.z + 1;
    for (int i = 0; i < armature_bone_count; i++)
    {
        int current_bone_bind = hull_table.z + i;
        int2 bone_bind_table = bone_bind_tables[current_bone_bind];

        float16 parent_transform = bone_bind_table.y == -1 
            ? model_transform 
            : armature_bones[bone_bind_table.y];

        float16 node_transform = get_node_transform(bone_bind_poses,
                                                    bone_channel_tables,
                                                    bone_pos_channel_tables,
                                                    bone_rot_channel_tables,
                                                    bone_scl_channel_tables,
                                                    animation_timing_indices,
                                                    animation_timings,
                                                    delta_time,
                                                    current_animation,
                                                    bone_bind_table.x);

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
