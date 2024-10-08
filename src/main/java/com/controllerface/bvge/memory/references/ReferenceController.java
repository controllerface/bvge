package com.controllerface.bvge.memory.references;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.crud.*;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.ReferenceContainer;
import com.controllerface.bvge.memory.groups.ReferenceBufferGroup;

import static com.controllerface.bvge.gpu.GPU.CL.arg_float2;

public class ReferenceController implements ReferenceContainer
{
    private final ReferenceBufferGroup reference_buffers;

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
    private final GPUKernel k_set_bone_channel_table;

    private int vertex_ref_index = 0;
    private int bone_bind_index = 0;
    private int bone_ref_index = 0;
    private int model_transform_index = 0;
    private int mesh_index = 0;
    private int face_index = 0;
    private int uv_index = 0;
    private int keyframe_index = 0;
    private int bone_channel_index = 0;
    private int animation_index = 0;

    public ReferenceController(CL_CommandQueue cmd_queue, GPUProgram p_gpu_crud, ReferenceBufferGroup reference_buffers)
    {
        this.reference_buffers = reference_buffers;

        k_create_texture_uv        = new CreateTextureUV_k(cmd_queue, p_gpu_crud).init(reference_buffers);
        k_create_keyframe          = new CreateKeyFrame_k(cmd_queue, p_gpu_crud).init(reference_buffers);
        k_create_vertex_reference  = new CreateVertexRef_k(cmd_queue, p_gpu_crud).init(reference_buffers);
        k_create_bone_bind_pose    = new CreateBoneBindPose_k(cmd_queue, p_gpu_crud).init(reference_buffers);
        k_create_bone_reference    = new CreateBoneRef_k(cmd_queue, p_gpu_crud).init(reference_buffers);
        k_create_bone_channel      = new CreateBoneChannel_k(cmd_queue, p_gpu_crud).init(reference_buffers);
        k_create_model_transform   = new CreateModelTransform_k(cmd_queue, p_gpu_crud).init(reference_buffers);
        k_create_mesh_reference    = new CreateMeshReference_k(cmd_queue, p_gpu_crud).init(reference_buffers);
        k_create_mesh_face         = new CreateMeshFace_k(cmd_queue, p_gpu_crud).init(reference_buffers);
        k_create_animation_timings = new CreateAnimationTimings_k(cmd_queue, p_gpu_crud).init(reference_buffers);
        k_set_bone_channel_table   = new SetBoneChannelTable_k(cmd_queue, p_gpu_crud).init(reference_buffers);
    }

    @Override
    public int next_mesh()
    {
        return mesh_index;
    }

    @Override
    public int new_vertex_reference(float x, float y, float[] weights, int[] uv_table)
    {
        int capacity = vertex_ref_index + 1;
        reference_buffers.ensure_vertex_reference(capacity);

        k_create_vertex_reference
            .set_arg(CreateVertexRef_k.Args.target, vertex_ref_index)
            .set_arg(CreateVertexRef_k.Args.new_vertex_reference, arg_float2(x, y))
            .set_arg(CreateVertexRef_k.Args.new_vertex_weights, weights)
            .set_arg(CreateVertexRef_k.Args.new_uv_table, uv_table)
            .call_task();

        return vertex_ref_index++;
    }

    @Override
    public int new_bone_bind_pose(float[] bone_data, int bone_layer)
    {
        int capacity = bone_bind_index + 1;
        reference_buffers.ensure_bind_pose(capacity);

        k_create_bone_bind_pose
            .set_arg(CreateBoneBindPose_k.Args.target, bone_bind_index)
            .set_arg(CreateBoneBindPose_k.Args.new_bone_bind_pose, bone_data)
            .set_arg(CreateBoneBindPose_k.Args.new_bone_layer, bone_layer)
            .call_task();

        return bone_bind_index++;
    }

    @Override
    public int new_bone_reference(float[] bone_data)
    {
        int capacity = bone_ref_index + 1;
        reference_buffers.ensure_bone_reference(capacity);

        k_create_bone_reference
            .set_arg(CreateBoneRef_k.Args.target, bone_ref_index)
            .set_arg(CreateBoneRef_k.Args.new_bone_reference, bone_data)
            .call_task();

        return bone_ref_index++;
    }

    @Override
    public int new_animation_timings(float duration, float tick_rate)
    {
        int capacity = animation_index + 1;

        reference_buffers.ensure_animation_timings(capacity);

        k_create_animation_timings
            .set_arg(CreateAnimationTimings_k.Args.target, animation_index)
            .set_arg(CreateAnimationTimings_k.Args.new_animation_duration, duration)
            .set_arg(CreateAnimationTimings_k.Args.new_animation_tick_rate, tick_rate)
            .call_task();

        return animation_index++;
    }

    @Override
    public int new_bone_channel(int anim_timing_index, int[] pos_table, int[] rot_table, int[] scl_table)
    {
        int capacity = bone_channel_index + 1;
        reference_buffers.ensure_bone_channel(capacity);

        k_create_bone_channel
            .set_arg(CreateBoneChannel_k.Args.target, bone_channel_index)
            .set_arg(CreateBoneChannel_k.Args.new_animation_timing_index, anim_timing_index)
            .set_arg(CreateBoneChannel_k.Args.new_bone_pos_channel_table, pos_table)
            .set_arg(CreateBoneChannel_k.Args.new_bone_rot_channel_table, rot_table)
            .set_arg(CreateBoneChannel_k.Args.new_bone_scl_channel_table, scl_table)
            .call_task();

        return bone_channel_index++;
    }

    @Override
    public int new_keyframe(float[] frame, float time)
    {
        int capacity = keyframe_index + 1;
        reference_buffers.ensure_keyframe(capacity);

        k_create_keyframe
            .set_arg(CreateKeyFrame_k.Args.target, keyframe_index)
            .set_arg(CreateKeyFrame_k.Args.new_keyframe, frame)
            .set_arg(CreateKeyFrame_k.Args.new_frame_time, time)
            .call_task();

        return keyframe_index++;
    }

    @Override
    public int new_texture_uv(float u, float v)
    {
        int capacity = uv_index + 1;
        reference_buffers.ensure_vertex_texture_uv(capacity);

        k_create_texture_uv
            .set_arg(CreateTextureUV_k.Args.target, uv_index)
            .set_arg(CreateTextureUV_k.Args.new_texture_uv, arg_float2(u, v))
            .call_task();

        return uv_index++;
    }

    @Override
    public int new_mesh_reference(int[] vertex_table, int[] face_table)
    {
        int capacity = mesh_index + 1;
        reference_buffers.ensure_mesh(capacity);

        k_create_mesh_reference
            .set_arg(CreateMeshReference_k.Args.target, mesh_index)
            .set_arg(CreateMeshReference_k.Args.new_mesh_vertex_table, vertex_table)
            .set_arg(CreateMeshReference_k.Args.new_mesh_face_table, face_table)
            .call_task();

        return mesh_index++;
    }

    @Override
    public int new_mesh_face(int[] face)
    {
        int capacity = face_index + 1;
        reference_buffers.ensure_mesh_face(capacity);

        k_create_mesh_face
            .set_arg(CreateMeshFace_k.Args.target, face_index)
            .set_arg(CreateMeshFace_k.Args.new_mesh_face, face)
            .call_task();

        return face_index++;
    }

    @Override
    public int new_model_transform(float[] transform_data)
    {
        int capacity = model_transform_index + 1;
        reference_buffers.ensure_model_transform(capacity);

        k_create_model_transform
            .set_arg(CreateModelTransform_k.Args.target, model_transform_index)
            .set_arg(CreateModelTransform_k.Args.new_model_transform, transform_data)
            .call_task();

        return model_transform_index++;
    }

    @Override
    public void set_bone_channel_table(int bind_pose_target, int[] channel_table)
    {
        k_set_bone_channel_table
            .set_arg(SetBoneChannelTable_k.Args.target, bind_pose_target)
            .set_arg(SetBoneChannelTable_k.Args.new_bone_channel_table, channel_table)
            .call_task();
    }
}
