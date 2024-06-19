package com.controllerface.bvge.game.world.sectors;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.cl.buffers.ReferenceGroup;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.crud.*;
import com.controllerface.bvge.cl.programs.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.arg_float2;
import static com.controllerface.bvge.cl.buffers.BufferType.*;
import static com.controllerface.bvge.cl.buffers.BufferType.ANIM_TICK_RATE;

public class ReferenceInput
{
    private final ReferenceGroup reference_group;

    private final GPUKernel k_create_animation_timings;
    private final GPUKernel k_create_bone_bind_pose;
    private final GPUKernel k_create_bone_channel;
    private final GPUKernel k_create_bone_reference;
    private final GPUKernel k_create_keyframe;
    private final GPUKernel k_create_mesh_face;
    private final GPUKernel k_create_mesh_reference;
    private final GPUKernel k_create_model_transform;
    private final GPUKernel k_create_texture_uv;
    private final GPUKernel k_create_vertex_reference;

    private int vertex_ref_index      = 0;
    private int bone_bind_index       = 0;
    private int bone_ref_index        = 0;
    private int model_transform_index = 0;
    private int mesh_index            = 0;
    private int face_index            = 0;
    private int uv_index              = 0;
    private int keyframe_index        = 0;
    private int bone_channel_index    = 0;
    private int animation_index       = 0;

    public ReferenceInput(long ptr_queue, GPUProgram p_gpu_crud, ReferenceGroup reference_group)
    {
        this.reference_group = reference_group;

        long k_ptr_create_texture_uv = p_gpu_crud.kernel_ptr(Kernel.create_texture_uv);
        k_create_texture_uv = new CreateTextureUV_k(ptr_queue, k_ptr_create_texture_uv)
                .buf_arg(CreateTextureUV_k.Args.texture_uvs, this.reference_group.get_buffer(VERTEX_TEXTURE_UV));

        long k_ptr_create_keyframe = p_gpu_crud.kernel_ptr(Kernel.create_keyframe);
        k_create_keyframe = new CreateKeyFrame_k(ptr_queue, k_ptr_create_keyframe)
                .buf_arg(CreateKeyFrame_k.Args.key_frames, this.reference_group.get_buffer(ANIM_KEY_FRAME))
                .buf_arg(CreateKeyFrame_k.Args.frame_times, this.reference_group.get_buffer(ANIM_FRAME_TIME));

        long k_ptr_create_vertex_reference = p_gpu_crud.kernel_ptr(Kernel.create_vertex_reference);
        k_create_vertex_reference = new CreateVertexRef_k(ptr_queue, k_ptr_create_vertex_reference)
                .buf_arg(CreateVertexRef_k.Args.vertex_references, this.reference_group.get_buffer(VERTEX_REFERENCE))
                .buf_arg(CreateVertexRef_k.Args.vertex_weights, this.reference_group.get_buffer(VERTEX_WEIGHT))
                .buf_arg(CreateVertexRef_k.Args.uv_tables, this.reference_group.get_buffer(VERTEX_UV_TABLE));

        long k_ptr_create_bone_bind_pose = p_gpu_crud.kernel_ptr(Kernel.create_bone_bind_pose);
        k_create_bone_bind_pose = new CreateBoneBindPose_k(ptr_queue, k_ptr_create_bone_bind_pose)
                .buf_arg(CreateBoneBindPose_k.Args.bone_bind_poses, this.reference_group.get_buffer(BONE_BIND_POSE));

        long k_ptr_create_bone_reference = p_gpu_crud.kernel_ptr(Kernel.create_bone_reference);
        k_create_bone_reference = new CreateBoneRef_k(ptr_queue, k_ptr_create_bone_reference)
                .buf_arg(CreateBoneRef_k.Args.bone_references, this.reference_group.get_buffer(BONE_REFERENCE));

        long k_ptr_create_bone_channel = p_gpu_crud.kernel_ptr(Kernel.create_bone_channel);
        k_create_bone_channel = new CreateBoneChannel_k(ptr_queue, k_ptr_create_bone_channel)
                .buf_arg(CreateBoneChannel_k.Args.animation_timing_indices, this.reference_group.get_buffer(ANIM_TIMING_INDEX))
                .buf_arg(CreateBoneChannel_k.Args.bone_pos_channel_tables, this.reference_group.get_buffer(ANIM_POS_CHANNEL))
                .buf_arg(CreateBoneChannel_k.Args.bone_rot_channel_tables, this.reference_group.get_buffer(ANIM_ROT_CHANNEL))
                .buf_arg(CreateBoneChannel_k.Args.bone_scl_channel_tables, this.reference_group.get_buffer(ANIM_SCL_CHANNEL));

        long k_ptr_create_model_transform = p_gpu_crud.kernel_ptr(Kernel.create_model_transform);
        k_create_model_transform = new CreateModelTransform_k(ptr_queue, k_ptr_create_model_transform)
                .buf_arg(CreateModelTransform_k.Args.model_transforms, this.reference_group.get_buffer(MODEL_TRANSFORM));

        long k_ptr_create_mesh_reference = p_gpu_crud.kernel_ptr(Kernel.create_mesh_reference);
        k_create_mesh_reference = new CreateMeshReference_k(ptr_queue, k_ptr_create_mesh_reference)
                .buf_arg(CreateMeshReference_k.Args.mesh_vertex_tables, this.reference_group.get_buffer(MESH_VERTEX_TABLE))
                .buf_arg(CreateMeshReference_k.Args.mesh_face_tables, this.reference_group.get_buffer(MESH_FACE_TABLE));

        long k_ptr_create_mesh_face = p_gpu_crud.kernel_ptr(Kernel.create_mesh_face);
        k_create_mesh_face = new CreateMeshFace_k(ptr_queue, k_ptr_create_mesh_face)
                .buf_arg(CreateMeshFace_k.Args.mesh_faces, this.reference_group.get_buffer(MESH_FACE));

        long k_ptr_create_animation_timings = p_gpu_crud.kernel_ptr(Kernel.create_animation_timings);
        k_create_animation_timings = new CreateAnimationTimings_k(ptr_queue, k_ptr_create_animation_timings)
                .buf_arg(CreateAnimationTimings_k.Args.animation_durations, this.reference_group.get_buffer(ANIM_DURATION))
                .buf_arg(CreateAnimationTimings_k.Args.animation_tick_rates, this.reference_group.get_buffer(ANIM_TICK_RATE));
    }

    public int mesh_index()
    {
        return mesh_index;
    }

    public int new_vertex_reference(float x, float y, float[] weights, int[] uv_table)
    {
        int capacity = vertex_ref_index + 1;
        reference_group.ensure_vertex_reference(capacity);

        k_create_vertex_reference
                .set_arg(CreateVertexRef_k.Args.target, vertex_ref_index)
                .set_arg(CreateVertexRef_k.Args.new_vertex_reference, arg_float2(x, y))
                .set_arg(CreateVertexRef_k.Args.new_vertex_weights, weights)
                .set_arg(CreateVertexRef_k.Args.new_uv_table, uv_table)
                .call(GPGPU.global_single_size);

        return vertex_ref_index++;
    }

    public int new_bone_bind_pose(float[] bone_data)
    {
        int capacity = bone_bind_index + 1;
        reference_group.ensure_bind_pose(capacity);

        k_create_bone_bind_pose
                .set_arg(CreateBoneBindPose_k.Args.target,bone_bind_index)
                .set_arg(CreateBoneBindPose_k.Args.new_bone_bind_pose, bone_data)
                .call(GPGPU.global_single_size);

        return bone_bind_index++;
    }

    public int new_bone_reference(float[] bone_data)
    {
        int capacity = bone_ref_index + 1;
        reference_group.ensure_bone_reference(capacity);

        k_create_bone_reference
                .set_arg(CreateBoneRef_k.Args.target, bone_ref_index)
                .set_arg(CreateBoneRef_k.Args.new_bone_reference, bone_data)
                .call(GPGPU.global_single_size);

        return bone_ref_index++;
    }

    public int new_animation_timings(float duration, float tick_rate)
    {
        int capacity = animation_index + 1;

        reference_group.ensure_animation_timings(capacity);

        k_create_animation_timings
                .set_arg(CreateAnimationTimings_k.Args.target, animation_index)
                .set_arg(CreateAnimationTimings_k.Args.new_animation_duration, duration)
                .set_arg(CreateAnimationTimings_k.Args.new_animation_tick_rate, tick_rate)
                .call(GPGPU.global_single_size);

        return animation_index++;
    }

    public int new_bone_channel(int anim_timing_index, int[] pos_table, int[] rot_table, int[] scl_table)
    {
        int capacity = bone_channel_index + 1;
        reference_group.ensure_bone_channel(capacity);

        k_create_bone_channel
                .set_arg(CreateBoneChannel_k.Args.target, bone_channel_index)
                .set_arg(CreateBoneChannel_k.Args.new_animation_timing_index, anim_timing_index)
                .set_arg(CreateBoneChannel_k.Args.new_bone_pos_channel_table, pos_table)
                .set_arg(CreateBoneChannel_k.Args.new_bone_rot_channel_table, rot_table)
                .set_arg(CreateBoneChannel_k.Args.new_bone_scl_channel_table, scl_table)
                .call(GPGPU.global_single_size);

        return bone_channel_index++;
    }

    public int new_keyframe(float[] frame, float time)
    {
        int capacity = keyframe_index + 1;
        reference_group.ensure_keyframe(capacity);

        k_create_keyframe
                .set_arg(CreateKeyFrame_k.Args.target, keyframe_index)
                .set_arg(CreateKeyFrame_k.Args.new_keyframe, frame)
                .set_arg(CreateKeyFrame_k.Args.new_frame_time, time)
                .call(GPGPU.global_single_size);

        return keyframe_index++;
    }

    public int new_texture_uv(float u, float v)
    {
        int capacity = uv_index + 1;
        reference_group.ensure_vertex_texture_uv(capacity);

        k_create_texture_uv
                .set_arg(CreateTextureUV_k.Args.target, uv_index)
                .set_arg(CreateTextureUV_k.Args.new_texture_uv, arg_float2(u, v))
                .call(GPGPU.global_single_size);

        return uv_index++;
    }

    public int new_mesh_reference(int[] vertex_table, int[] face_table)
    {
        int capacity = mesh_index + 1;
        reference_group.ensure_mesh(capacity);

        k_create_mesh_reference
                .set_arg(CreateMeshReference_k.Args.target, mesh_index)
                .set_arg(CreateMeshReference_k.Args.new_mesh_vertex_table, vertex_table)
                .set_arg(CreateMeshReference_k.Args.new_mesh_face_table, face_table)
                .call(GPGPU.global_single_size);

        return mesh_index++;
    }

    public int new_mesh_face(int[] face)
    {
        int capacity = face_index + 1;
        reference_group.ensure_mesh_face(capacity);

        k_create_mesh_face
                .set_arg(CreateMeshFace_k.Args.target, face_index)
                .set_arg(CreateMeshFace_k.Args.new_mesh_face, face)
                .call(GPGPU.global_single_size);

        return face_index++;
    }

    public int new_model_transform(float[] transform_data)
    {
        int capacity = model_transform_index + 1;
        reference_group.ensure_model_transform(capacity);

        k_create_model_transform
                .set_arg(CreateModelTransform_k.Args.target, model_transform_index)
                .set_arg(CreateModelTransform_k.Args.new_model_transform, transform_data)
                .call(GPGPU.global_single_size);

        return model_transform_index++;
    }
}