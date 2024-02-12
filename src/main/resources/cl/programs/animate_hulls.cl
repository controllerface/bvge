constant float16 identity_matrix = (float16)
(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
);

inline float16 translation_vector_to_matrix(float4 vector)
{
    float16 matrix;
    matrix.s0 = 1;
    matrix.s1 = 0;
    matrix.s2 = 0;
    matrix.s3 = 0;
    matrix.s4 = 0;
    matrix.s5 = 1;
    matrix.s6 = 0;
    matrix.s7 = 0;
    matrix.s8 = 0;
    matrix.s9 = 0;
    matrix.sA = 1;
    matrix.sB = 0;
    matrix.sC = vector.x;
    matrix.sD = vector.y;
    matrix.sE = vector.z;
    matrix.sF = vector.w;
    return matrix;
}

inline float16 scaling_vector_to_matrix(float4 vector)
{
    float16 matrix;
    matrix.s0 = vector.x;
    matrix.s1 = 0;
    matrix.s2 = 0;
    matrix.s3 = 0;

    matrix.s4 = 0;
    matrix.s5 = vector.y;
    matrix.s6 = 0;
    matrix.s7 = 0;

    matrix.s8 = 0;
    matrix.s9 = 0;
    matrix.sA = vector.z;
    matrix.sB = 0;

    matrix.sC = 0;
    matrix.sD = 0;
    matrix.sE = 0;
    matrix.sF = vector.w;
    return matrix;
}

inline float16 rotation_quaternion_to_matrix(float4 quaternion)
{
    float16 matrix;
    float w2 = quaternion.w * quaternion.w;
    float x2 = quaternion.x * quaternion.x;
    float y2 = quaternion.y * quaternion.y;
    float z2 = quaternion.z * quaternion.z;
    float zw = quaternion.z * quaternion.w;
    float xy = quaternion.x * quaternion.y;
    float xz = quaternion.x * quaternion.z;
    float yw = quaternion.y * quaternion.w;
    float yz = quaternion.y * quaternion.z;
    float xw = quaternion.x * quaternion.w;
    matrix.s0 = w2 + x2 - z2 - y2;
    matrix.s1 = xy + zw + zw + xy;
    matrix.s2 = xz - yw + xz - yw;
    matrix.s3 = 0.0F;
    matrix.s4 = -zw + xy - zw + xy;
    matrix.s5 = y2 - z2 + w2 - x2;
    matrix.s6 = yz + yz + xw + xw;
    matrix.s7 = 0.0F;
    matrix.s8 = yw + xz + xz + yw;
    matrix.s9 = yz + yz - xw - xw;
    matrix.sA = z2 - y2 - x2 + w2;
    matrix.sB = 0.0F;
    matrix.sC = 0.0F;
    matrix.sD = 0.0F;
    matrix.sE = 0.0F;
    matrix.sF = 1.0F;
    return matrix;

    // float w = quaternion.x;
    // float x = quaternion.y;
    // float y = quaternion.z;
    // float z = quaternion.w;

    // float xx = x * x;
    // float xy = x * y;
    // float xz = x * z;
    // float xw = x * w;

    // float yy = y * y;
    // float yz = y * z;
    // float yw = y * w;

    // float zz = z * z;
    // float zw = z * w;


    // matrix.s0 = 1 - 2 * (yy + zz);
    // matrix.s1 = 2 * (xy - zw);
    // matrix.s2 = 2 * (xz + yw);
    // matrix.s3 = 0;
    // matrix.s4 = 2 * (xy + zw);
    // matrix.s5 = 1 - 2 * (xx + zz);
    // matrix.s6 = 2 * (yz - xw);
    // matrix.s7 = 0;
    // matrix.s8 = 2 * (xz - yw);
    // matrix.s9 = 2 * (yz + xw);
    // matrix.sA = 1 - 2 * (xx + yy);
    // matrix.sB = 0;
    // matrix.sC = 0;
    // matrix.sD = 0;
    // matrix.sE = 0;
    // matrix.sF = 1;

    // return matrix;
}

inline float4 vector_lerp(float4 a, float4 b, float t) 
{
    return fma(b - a, t, a);
}

inline float4 quaternion_lerp(float4 a, float4 b, float factor) 
{
    float4 dest;
    float cosom = fma(a.x, b.x, fma(a.y, b.y, fma(a.z, b.z, a.w * b.w)));
    float scale0 = 1.0F - factor;
    float scale1 = cosom >= 0.0F ? factor : -factor;
    
    dest.x = fma(scale0, a.x, scale1 * b.x);
    dest.y = fma(scale0, a.y, scale1 * b.y);
    dest.z = fma(scale0, a.z, scale1 * b.z);
    dest.w = fma(scale0, a.w, scale1 * b.w);

    float s = rsqrt(fma(dest.x, dest.x, fma(dest.y, dest.y, fma(dest.z, dest.z, dest.w * dest.w))));
    
    dest.x *= s;
    dest.y *= s;
    dest.z *= s;
    dest.w *= s;
    
    return dest;
}


// inline float4 quaternion_lerp(float4 a, float4 b, float t) 
// {
//     float4 result;

//     // Perform linear interpolation for each quaternion component individually
//     result.x = (1 - t) * a.x + t * b.x;
//     result.y = (1 - t) * a.y + t * b.y;
//     result.z = (1 - t) * a.z + t * b.z;
//     result.w = (1 - t) * a.w + t * b.w;

//     // Normalize the interpolated quaternion
//     float norm = (float) sqrt(result.x * result.x + result.y * result.y +
//                                 result.z * result.z + result.w * result.w);
//     if (norm != 0) 
//     {
//         result.x /= norm;
//         result.y /= norm;
//         result.z /= norm;
//         result.w /= norm;
//     }

//     return result;
// }

float16 get_node_transform(__global float16 *bone_bind_poses,
                                  __global int2 *bone_channel_tables,
                                  __global int2 *bone_pos_channel_tables,
                                  __global int2 *bone_rot_channel_tables,
                                  __global int2 *bone_scl_channel_tables,
                                  __global int *animation_timing_indices,
                                  __global double2 *animation_timings,
                                  __global float4 *key_frames,
                                  __global double *frame_times,
                                  double current_time,
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
    double2 timings = animation_timings[timing_index];

    double time_in_ticks = current_time * timings.y;
    double anim_time_ticks = fmod(time_in_ticks, timings.x);

    int pos_count = pos_channel_table.y - pos_channel_table.x + 1;
    int rot_count = rot_channel_table.y - rot_channel_table.x + 1;
    int scl_count = scl_channel_table.y - scl_channel_table.x + 1;

    float4 pos_a;
    float4 pos_b;
    float pos_factor;

    float4 rot_a;
    float4 rot_b;
    float rot_factor;

    float4 scl_a;
    float4 scl_b;
    float scl_factor;

    for (int pos_index_b = pos_channel_table.x; pos_index_b <= pos_channel_table.y; pos_index_b++)
    {
        double time_b = frame_times[pos_index_b];
        if (time_b > anim_time_ticks)
        {
            int pos_index_a = pos_index_b - 1;
            pos_a = key_frames[pos_index_a];
            pos_b = key_frames[pos_index_b];
            
            double time_a = frame_times[pos_index_a];
            float delta = (float)time_b - (float)time_a;
            pos_factor = ((float)anim_time_ticks - (float)time_a) / delta;
            break;
        }
    }

    for (int rot_index_b = rot_channel_table.x; rot_index_b <= rot_channel_table.y; rot_index_b++)
    {
        double time_b = frame_times[rot_index_b];
        if (time_b > anim_time_ticks)
        {
            int rot_index_a = rot_index_b - 1;
            rot_a = key_frames[rot_index_a];
            rot_b = key_frames[rot_index_b];

            double time_a = frame_times[rot_index_a];
            float delta = (float)time_b - (float)time_a;
            rot_factor = ((float)anim_time_ticks - (float)time_a) / delta;
            break;
        }
    }

    for (int scl_index_b = scl_channel_table.x; scl_index_b <= scl_channel_table.y; scl_index_b++)
    {
        double time_b = frame_times[scl_index_b];
        if (time_b > anim_time_ticks)
        {
            int scl_index_a = scl_index_b - 1;
            scl_a = key_frames[scl_index_a];
            scl_b = key_frames[scl_index_b];

            double time_a = frame_times[scl_index_a];
            float delta = (float)time_b - (float)time_a;
            scl_factor = ((float)anim_time_ticks - (float)time_a) / delta;
            break;
        }
    }

    float4 pos_final = vector_lerp(pos_a, pos_b, pos_factor);
    float4 rot_final = quaternion_lerp(rot_a, rot_b, rot_factor);
    float4 scl_final = vector_lerp(scl_a, scl_b, scl_factor);

    float16 pos_matrix = translation_vector_to_matrix(pos_final);
    float16 rot_matrix = rotation_quaternion_to_matrix(rot_final);
    float16 scl_matrix = scaling_vector_to_matrix(scl_final);

    float16 fmat_1 = matrix_mul(pos_matrix, rot_matrix);
    float16 fmat_2 = matrix_mul(fmat_1, scl_matrix);
    //float16 fmat_3 = matrix_mul_affine(fmat_2, bone_bind_poses[bone_id]);

    //float16 expected = bone_bind_poses[bone_id];
    //if (bone_id == 0)
    //{
        // printf("debug e: id=%d \n e.s0: %f e.s1: %f e.s2: %f e.s3: %f \n e.s4: %f e.s5: %f e.s6: %f e.s7: %f \n e.s8: %f e.s9: %f e.sA: %f e.sB: %f \n e.sC: %f e.sD: %f e.sE: %f e.sF: %f", 
        //     bone_id,
        //     expected.s0, expected.s1, expected.s2, expected.s3,
        //     expected.s4, expected.s5, expected.s6, expected.s7,
        //     expected.s8, expected.s9, expected.sA, expected.sB,
        //     expected.sC, expected.sD, expected.sE, expected.sF);

        // printf("debug a: id=%d \n a.s0: %f a.s1: %f a.s2: %f a.s3: %f \n a.s4: %f a.s5: %f a.s6: %f a.s7: %f \n a.s8: %f a.s9: %f a.sA: %f a.sB: %f \n a.sC: %f a.sD: %f a.sE: %f a.sF: %f",
        //     bone_id,
        //     fmat_2.s0, fmat_2.s1, fmat_2.s2, fmat_2.s3,
        //     fmat_2.s4, fmat_2.s5, fmat_2.s6, fmat_2.s7,
        //     fmat_2.s8, fmat_2.s9, fmat_2.sA, fmat_2.sB,
        //     fmat_2.sC, fmat_2.sD, fmat_2.sE, fmat_2.sF);
    //}

    return fmat_2;
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
                                __global float4 *key_frames,
                                __global double *frame_times,
                                __global int *animation_timing_indices,
                                __global double2 *animation_timings,
                                __global int *armature_animation_indices,
                                __global double *armature_animation_elapsed,
                                float delta_time)
{
    int current_armature = get_global_id(0);
    int4 hull_table = hull_tables[current_armature];
    int4 armature_flag = armature_flags[current_armature];
    float16 model_transform = model_transforms[armature_flag.w];
    int current_animation = armature_animation_indices[current_armature]; 
    double current_frame_time = armature_animation_elapsed[current_armature] += (double)delta_time;

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
                                                    key_frames,
                                                    frame_times,
                                                    current_frame_time,
                                                    current_animation,
                                                    bone_bind_table.x);

        float16 global_transform = matrix_mul_affine(parent_transform, node_transform);
        armature_bones[current_bone_bind] = global_transform;
    }
    armature_animation_elapsed[current_armature] = current_frame_time;
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
