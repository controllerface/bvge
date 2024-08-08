package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.crud.*;

public class GPUCrud extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(const_entity_flags);
        src.add(const_hull_flags);
        src.add(func_rotate_point);
        src.add(CreatePoint_k.kernel_source);
        src.add(CreateEdge_k.kernel_source);
        src.add(CreateHull_k.kernel_source);
        src.add(CreateEntity_k.kernel_source);
        src.add(CreateBoneChannel_k.kernel_source);
        src.add(CreateAnimationTimings_k.kernel_source);
        src.add(CreateKeyFrame_k.kernel_source);
        src.add(CreateTextureUV_k.kernel_source);
        src.add(CreateVertexRef_k.kernel_source);
        src.add(CreateModelTransform_k.kernel_source);
        src.add(CreateBoneBindPose_k.kernel_source);
        src.add(CreateBoneRef_k.kernel_source);
        src.add(CreateHullBone_k.kernel_source);
        src.add(CreateEntityBone_k.kernel_source);
        src.add(CreateMeshReference_k.kernel_source);
        src.add(CreateMeshFace_k.kernel_source);
        src.add(SetBoneChannelTable_k.kernel_source);
        src.add(GPU.CL.read_src("programs/gpu_crud.cl"));

        make_program();

        load_kernel(KernelType.create_animation_timings);
        load_kernel(KernelType.create_entity);
        load_kernel(KernelType.create_entity_bone);
        load_kernel(KernelType.create_hull_bone);
        load_kernel(KernelType.create_bone_bind_pose);
        load_kernel(KernelType.create_bone_channel);
        load_kernel(KernelType.create_bone_reference);
        load_kernel(KernelType.create_edge);
        load_kernel(KernelType.create_hull);
        load_kernel(KernelType.create_keyframe);
        load_kernel(KernelType.create_mesh_face);
        load_kernel(KernelType.create_mesh_reference);
        load_kernel(KernelType.create_model_transform);
        load_kernel(KernelType.create_point);
        load_kernel(KernelType.create_texture_uv);
        load_kernel(KernelType.create_vertex_reference);
        load_kernel(KernelType.count_egress_entities);
        load_kernel(KernelType.egress_entities);
        load_kernel(KernelType.egress_hulls);
        load_kernel(KernelType.egress_edges);
        load_kernel(KernelType.egress_points);
        load_kernel(KernelType.egress_hull_bones);
        load_kernel(KernelType.egress_entity_bones);
        load_kernel(KernelType.egress_broken);
        load_kernel(KernelType.egress_collected);
        load_kernel(KernelType.merge_point);
        load_kernel(KernelType.merge_edge);
        load_kernel(KernelType.merge_hull);
        load_kernel(KernelType.merge_entity);
        load_kernel(KernelType.merge_hull_bone);
        load_kernel(KernelType.merge_entity_bone);
        load_kernel(KernelType.read_position);
        load_kernel(KernelType.read_entity_info);
        load_kernel(KernelType.write_entity_info);
        load_kernel(KernelType.rotate_hull);
        load_kernel(KernelType.set_bone_channel_table);
        load_kernel(KernelType.update_accel);
        load_kernel(KernelType.update_mouse_position);
        load_kernel(KernelType.update_block_position);
        load_kernel(KernelType.update_select_block);
        load_kernel(KernelType.clear_select_block);
        load_kernel(KernelType.place_block);

        return this;
    }
}
