typedef struct 
{
    float4 frame_a;
    float4 frame_b;
    float lerp_factor;
} KeyFramePair;

inline KeyFramePair find_keyframe_pair(__global float4 *key_frames, 
                                       __global float *frame_times,
                                       float anim_time_ticks,
                                       int2 channel_table)
{
    KeyFramePair result;
    for (int pos_index_b = channel_table.x; pos_index_b <= channel_table.y; pos_index_b++)
    {
        float time_b = frame_times[pos_index_b];
        if (time_b > anim_time_ticks)
        {
            int pos_index_a = pos_index_b - 1;
            result.frame_a = key_frames[pos_index_a];
            result.frame_b = key_frames[pos_index_b];
            
            float time_a = frame_times[pos_index_a];
            float delta = time_b - time_a;
            result.lerp_factor =  native_divide( (anim_time_ticks - time_a), delta);
            break;
        }
    }
    return result;
}

float16 get_node_transform(__global float16 *bone_bind_poses,
                           __global int2 *bone_channel_tables,
                           __global int2 *bone_pos_channel_tables,
                           __global int2 *bone_rot_channel_tables,
                           __global int2 *bone_scl_channel_tables,
                           __global int *animation_timing_indices,
                           __global float *animation_durations,
                           __global float *animation_tick_rates,
                           __global float4 *key_frames,
                           __global float *frame_times,
                           float current_time,
                           int animation_index,
                           int bone_id)
{
    if (animation_index < 0) return bone_bind_poses[bone_id];

    int2 channel_table = bone_channel_tables[bone_id];
    int channel_index = channel_table.x - animation_index;
    int timing_index = animation_timing_indices[channel_index];
    int2 pos_channel_table = bone_pos_channel_tables[channel_index];
    int2 rot_channel_table = bone_rot_channel_tables[channel_index];
    int2 scl_channel_table = bone_scl_channel_tables[channel_index];
    float duration = animation_durations[timing_index];
    float tick_rate = animation_tick_rates[timing_index];

    float time_in_ticks = current_time * tick_rate;
    float anim_time_ticks = fmod(time_in_ticks, duration);

    KeyFramePair pos_pair = find_keyframe_pair(key_frames, frame_times, anim_time_ticks, pos_channel_table);
    KeyFramePair rot_pair = find_keyframe_pair(key_frames, frame_times, anim_time_ticks, rot_channel_table);
    KeyFramePair scl_pair = find_keyframe_pair(key_frames, frame_times, anim_time_ticks, scl_channel_table);

    float4 pos_final = vector_lerp(pos_pair.frame_a, pos_pair.frame_b, pos_pair.lerp_factor);
    float4 rot_final = quaternion_lerp(rot_pair.frame_a, rot_pair.frame_b, rot_pair.lerp_factor);
    float4 scl_final = vector_lerp(scl_pair.frame_a, scl_pair.frame_b, scl_pair.lerp_factor);

    float16 pos_matrix = translation_vector_to_matrix(pos_final);
    float16 rot_matrix = rotation_quaternion_to_matrix(rot_final);
    float16 scl_matrix = scaling_vector_to_matrix(scl_final);

    return matrix_mul(matrix_mul(pos_matrix, rot_matrix), scl_matrix);
}

__kernel void animate_armatures(__global float16 *armature_bones,
                                __global float16 *bone_bind_poses,
                                __global float16 *model_transforms,
                                __global int *armature_bone_reference_ids,
                                __global int *armature_bone_parent_ids,
                                __global int2 *bone_channel_tables,
                                __global int2 *bone_pos_channel_tables,
                                __global int2 *bone_rot_channel_tables,
                                __global int2 *bone_scl_channel_tables,
                                __global int *armature_model_transforms,
                                __global int2 *bone_tables,
                                __global float4 *key_frames,
                                __global float *frame_times,
                                __global int *animation_timing_indices,
                                __global float *animation_durations,
                                __global float *animation_tick_rates,
                                __global int *armature_animation_indices,
                                __global float *armature_animation_elapsed,
                                float delta_time)
{
    int current_armature = get_global_id(0);
    int2 bone_table = bone_tables[current_armature];
    int armature_transform_id = armature_model_transforms[current_armature];
    float16 model_transform = model_transforms[armature_transform_id];
    int current_animation = armature_animation_indices[current_armature]; 
    float current_frame_time = armature_animation_elapsed[current_armature] += delta_time;

    // note that armatures with no bones simply do nothing as the bone count will be zero
    int armature_bone_count = bone_table.y - bone_table.x + 1;
    for (int i = 0; i < armature_bone_count; i++)
    {
        int current_bone_bind = bone_table.x + i;
        int bone_reference_id = armature_bone_reference_ids[current_bone_bind];
        int bone_parent_id = armature_bone_parent_ids[current_bone_bind];

        float16 parent_transform = bone_parent_id == -1 
            ? model_transform 
            : armature_bones[bone_parent_id];

        float16 node_transform = get_node_transform(bone_bind_poses,
                                                    bone_channel_tables,
                                                    bone_pos_channel_tables,
                                                    bone_rot_channel_tables,
                                                    bone_scl_channel_tables,
                                                    animation_timing_indices,
                                                    animation_durations,
                                                    animation_tick_rates,
                                                    key_frames,
                                                    frame_times,
                                                    current_frame_time,
                                                    current_animation,
                                                    bone_reference_id);

        float16 global_transform = matrix_mul_affine(parent_transform, node_transform);
        armature_bones[current_bone_bind] = global_transform;
    }
    armature_animation_elapsed[current_armature] = current_frame_time;
}

__kernel void animate_bones(__global float16 *bones,
                            __global float16 *bone_references,
                            __global float16 *armature_bones,
                            __global int *hull_bind_pose_indicies,
                            __global int *hull_inv_bind_pose_indicies)
{
    int current_bone = get_global_id(0);
    int bind_pose_id = hull_bind_pose_indicies[current_bone];
    int inv_bind_pose_id = hull_inv_bind_pose_indicies[current_bone];
    float16 bone_reference = bone_references[inv_bind_pose_id];
    float16 armature_bone = armature_bones[bind_pose_id];
    float16 next_position = matrix_mul_affine(armature_bone, bone_reference);
    bones[current_bone] = next_position;
}

__kernel void animate_points(__global float4 *points,
                             __global float4 *hulls,
                             __global int *hull_armature_ids,
                             __global int *hull_flags,
                             __global int *point_vertex_references,
                             __global int *point_hull_indices,
                             __global int4 *bone_tables,
                             __global float4 *vertex_weights,
                             __global float4 *armatures,
                             __global float2 *vertex_references,
                             __global float16 *bones)
{

    int gid = get_global_id(0);
    float4 point = points[gid];

    int point_vertex_reference = point_vertex_references[gid];
    int point_hull_index = point_hull_indices[gid];
    
    int hull_flag = hull_flags[point_hull_index];
    int hull_armature_id = hull_armature_ids[point_hull_index];
    bool no_bones = (hull_flag & NO_BONES) !=0;
    if (no_bones) return;

    int4 bone_table = bone_tables[gid];

    float2 reference_vertex = vertex_references[point_vertex_reference];
    float4 reference_weights = vertex_weights[point_vertex_reference];

    float16 bone1 = bone_table.x == -1 ? identity_matrix : bones[bone_table.x];
    float16 bone2 = bone_table.y == -1 ? identity_matrix : bones[bone_table.y];
    float16 bone3 = bone_table.z == -1 ? identity_matrix : bones[bone_table.z];
    float16 bone4 = bone_table.w == -1 ? identity_matrix : bones[bone_table.w];

    float16 test_bone = bone1 * reference_weights.x;
    test_bone += bone2 * reference_weights.y;
    test_bone += bone3 * reference_weights.z;
    test_bone += bone4 * reference_weights.w;
    
    float4 hull = hulls[point_hull_index];
    float4 armature = armatures[hull_armature_id]; 

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
