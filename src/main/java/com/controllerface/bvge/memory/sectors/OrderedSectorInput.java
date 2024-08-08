package com.controllerface.bvge.memory.sectors;

import com.controllerface.bvge.gpu.GPUResource;
import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.crud.*;
import com.controllerface.bvge.gpu.cl.programs.GPUCrud;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.SectorContainer;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;
import com.controllerface.bvge.memory.types.CoreBufferType;

import static com.controllerface.bvge.gpu.GPU.CL.arg_long;

public class OrderedSectorInput implements SectorContainer, GPUResource
{
    private static final long ENTITY_INIT = 1_000L;
    private static final long HULL_INIT   = 1_000L;
    private static final long EDGE_INIT   = 2_400L;
    private static final long POINT_INIT  = 5_000L;

    private final GPUProgram p_gpu_crud;

    private final GPUKernel k_merge_point;
    private final GPUKernel k_merge_edge;
    private final GPUKernel k_merge_hull;
    private final GPUKernel k_merge_entity;
    private final GPUKernel k_merge_hull_bone;
    private final GPUKernel k_merge_entity_bone;

    private final CoreBufferGroup buffers;
    private final SectorController controller;

    public OrderedSectorInput(long ptr_queue, GPUCoreMemory core_memory)
    {
        this.p_gpu_crud = new GPUCrud().init();
        this.buffers    = new CoreBufferGroup("Sector Ingress", ptr_queue, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        this.controller = new SectorController(ptr_queue, this.p_gpu_crud, this.buffers);

        long k_ptr_merge_point = this.p_gpu_crud.kernel_ptr(KernelType.merge_point);
        k_merge_point = new MergePoint_k(ptr_queue, k_ptr_merge_point)
            .buf_arg(MergePoint_k.Args.points_in, buffers.buffer(CoreBufferType.POINT))
            .buf_arg(MergePoint_k.Args.point_vertex_references_in, buffers.buffer(CoreBufferType.POINT_VERTEX_REFERENCE))
            .buf_arg(MergePoint_k.Args.point_hull_indices_in, buffers.buffer(CoreBufferType.POINT_HULL_INDEX))
            .buf_arg(MergePoint_k.Args.point_hit_counts_in, buffers.buffer(CoreBufferType.POINT_HIT_COUNT))
            .buf_arg(MergePoint_k.Args.point_flags_in, buffers.buffer(CoreBufferType.POINT_FLAG))
            .buf_arg(MergePoint_k.Args.point_bone_tables_in, buffers.buffer(CoreBufferType.POINT_BONE_TABLE))
            .buf_arg(MergePoint_k.Args.points_out, core_memory.get_buffer(CoreBufferType.POINT))
            .buf_arg(MergePoint_k.Args.point_vertex_references_out, core_memory.get_buffer(CoreBufferType.POINT_VERTEX_REFERENCE))
            .buf_arg(MergePoint_k.Args.point_hull_indices_out, core_memory.get_buffer(CoreBufferType.POINT_HULL_INDEX))
            .buf_arg(MergePoint_k.Args.point_hit_counts_out, core_memory.get_buffer(CoreBufferType.POINT_HIT_COUNT))
            .buf_arg(MergePoint_k.Args.point_flags_out, core_memory.get_buffer(CoreBufferType.POINT_FLAG))
            .buf_arg(MergePoint_k.Args.point_bone_tables_out, core_memory.get_buffer(CoreBufferType.POINT_BONE_TABLE));

        long k_ptr_merge_edge = this.p_gpu_crud.kernel_ptr(KernelType.merge_edge);
        k_merge_edge = new MergeEdge_k(ptr_queue, k_ptr_merge_edge)
            .buf_arg(MergeEdge_k.Args.edges_in,         buffers.buffer(CoreBufferType.EDGE))
            .buf_arg(MergeEdge_k.Args.edge_lengths_in,  buffers.buffer(CoreBufferType.EDGE_LENGTH))
            .buf_arg(MergeEdge_k.Args.edge_flags_in,    buffers.buffer(CoreBufferType.EDGE_FLAG))
            .buf_arg(MergeEdge_k.Args.edge_pins_in,     buffers.buffer(CoreBufferType.EDGE_PIN))
            .buf_arg(MergeEdge_k.Args.edges_out,        core_memory.get_buffer(CoreBufferType.EDGE))
            .buf_arg(MergeEdge_k.Args.edge_lengths_out, core_memory.get_buffer(CoreBufferType.EDGE_LENGTH))
            .buf_arg(MergeEdge_k.Args.edge_flags_out,   core_memory.get_buffer(CoreBufferType.EDGE_FLAG))
            .buf_arg(MergeEdge_k.Args.edge_pins_out,    core_memory.get_buffer(CoreBufferType.EDGE_PIN));

        long k_ptr_merge_hull = this.p_gpu_crud.kernel_ptr(KernelType.merge_hull);
        k_merge_hull = new MergeHull_k(ptr_queue, k_ptr_merge_hull)
            .buf_arg(MergeHull_k.Args.hulls_in, buffers.buffer(CoreBufferType.HULL))
            .buf_arg(MergeHull_k.Args.hull_scales_in, buffers.buffer(CoreBufferType.HULL_SCALE))
            .buf_arg(MergeHull_k.Args.hull_rotations_in, buffers.buffer(CoreBufferType.HULL_ROTATION))
            .buf_arg(MergeHull_k.Args.hull_frictions_in, buffers.buffer(CoreBufferType.HULL_FRICTION))
            .buf_arg(MergeHull_k.Args.hull_restitutions_in, buffers.buffer(CoreBufferType.HULL_RESTITUTION))
            .buf_arg(MergeHull_k.Args.hull_point_tables_in, buffers.buffer(CoreBufferType.HULL_POINT_TABLE))
            .buf_arg(MergeHull_k.Args.hull_edge_tables_in, buffers.buffer(CoreBufferType.HULL_EDGE_TABLE))
            .buf_arg(MergeHull_k.Args.hull_bone_tables_in, buffers.buffer(CoreBufferType.HULL_BONE_TABLE))
            .buf_arg(MergeHull_k.Args.hull_entity_ids_in, buffers.buffer(CoreBufferType.HULL_ENTITY_ID))
            .buf_arg(MergeHull_k.Args.hull_flags_in, buffers.buffer(CoreBufferType.HULL_FLAG))
            .buf_arg(MergeHull_k.Args.hull_mesh_ids_in, buffers.buffer(CoreBufferType.HULL_MESH_ID))
            .buf_arg(MergeHull_k.Args.hull_uv_offsets_in, buffers.buffer(CoreBufferType.HULL_UV_OFFSET))
            .buf_arg(MergeHull_k.Args.hull_integrity_in, buffers.buffer(CoreBufferType.HULL_INTEGRITY))
            .buf_arg(MergeHull_k.Args.hulls_out, core_memory.get_buffer(CoreBufferType.HULL))
            .buf_arg(MergeHull_k.Args.hull_scales_out, core_memory.get_buffer(CoreBufferType.HULL_SCALE))
            .buf_arg(MergeHull_k.Args.hull_rotations_out, core_memory.get_buffer(CoreBufferType.HULL_ROTATION))
            .buf_arg(MergeHull_k.Args.hull_frictions_out, core_memory.get_buffer(CoreBufferType.HULL_FRICTION))
            .buf_arg(MergeHull_k.Args.hull_restitutions_out, core_memory.get_buffer(CoreBufferType.HULL_RESTITUTION))
            .buf_arg(MergeHull_k.Args.hull_point_tables_out, core_memory.get_buffer(CoreBufferType.HULL_POINT_TABLE))
            .buf_arg(MergeHull_k.Args.hull_edge_tables_out, core_memory.get_buffer(CoreBufferType.HULL_EDGE_TABLE))
            .buf_arg(MergeHull_k.Args.hull_bone_tables_out, core_memory.get_buffer(CoreBufferType.HULL_BONE_TABLE))
            .buf_arg(MergeHull_k.Args.hull_entity_ids_out, core_memory.get_buffer(CoreBufferType.HULL_ENTITY_ID))
            .buf_arg(MergeHull_k.Args.hull_flags_out, core_memory.get_buffer(CoreBufferType.HULL_FLAG))
            .buf_arg(MergeHull_k.Args.hull_mesh_ids_out, core_memory.get_buffer(CoreBufferType.HULL_MESH_ID))
            .buf_arg(MergeHull_k.Args.hull_uv_offsets_out, core_memory.get_buffer(CoreBufferType.HULL_UV_OFFSET))
            .buf_arg(MergeHull_k.Args.hull_integrity_out, core_memory.get_buffer(CoreBufferType.HULL_INTEGRITY));

        long k_ptr_merge_entity = this.p_gpu_crud.kernel_ptr(KernelType.merge_entity);
        k_merge_entity = new MergeEntity_k(ptr_queue, k_ptr_merge_entity)
            .buf_arg(MergeEntity_k.Args.entities_in, buffers.buffer(CoreBufferType.ENTITY))
            .buf_arg(MergeEntity_k.Args.entity_animation_time_in, buffers.buffer(CoreBufferType.ENTITY_ANIM_TIME))
            .buf_arg(MergeEntity_k.Args.entity_previous_time_in, buffers.buffer(CoreBufferType.ENTITY_PREV_TIME))
            .buf_arg(MergeEntity_k.Args.entity_motion_states_in, buffers.buffer(CoreBufferType.ENTITY_MOTION_STATE))
            .buf_arg(MergeEntity_k.Args.entity_animation_layers_in, buffers.buffer(CoreBufferType.ENTITY_ANIM_LAYER))
            .buf_arg(MergeEntity_k.Args.entity_previous_layers_in, buffers.buffer(CoreBufferType.ENTITY_PREV_LAYER))
            .buf_arg(MergeEntity_k.Args.entity_hull_tables_in, buffers.buffer(CoreBufferType.ENTITY_HULL_TABLE))
            .buf_arg(MergeEntity_k.Args.entity_bone_tables_in, buffers.buffer(CoreBufferType.ENTITY_BONE_TABLE))
            .buf_arg(MergeEntity_k.Args.entity_masses_in, buffers.buffer(CoreBufferType.ENTITY_MASS))
            .buf_arg(MergeEntity_k.Args.entity_root_hulls_in, buffers.buffer(CoreBufferType.ENTITY_ROOT_HULL))
            .buf_arg(MergeEntity_k.Args.entity_model_indices_in, buffers.buffer(CoreBufferType.ENTITY_MODEL_ID))
            .buf_arg(MergeEntity_k.Args.entity_model_transforms_in, buffers.buffer(CoreBufferType.ENTITY_TRANSFORM_ID))
            .buf_arg(MergeEntity_k.Args.entity_types_in, buffers.buffer(CoreBufferType.ENTITY_TYPE))
            .buf_arg(MergeEntity_k.Args.entity_flags_in, buffers.buffer(CoreBufferType.ENTITY_FLAG))
            .buf_arg(MergeEntity_k.Args.entities_out, core_memory.get_buffer(CoreBufferType.ENTITY))
            .buf_arg(MergeEntity_k.Args.entity_animation_time_out, core_memory.get_buffer(CoreBufferType.ENTITY_ANIM_TIME))
            .buf_arg(MergeEntity_k.Args.entity_previous_time_out, core_memory.get_buffer(CoreBufferType.ENTITY_PREV_TIME))
            .buf_arg(MergeEntity_k.Args.entity_motion_states_out, core_memory.get_buffer(CoreBufferType.ENTITY_MOTION_STATE))
            .buf_arg(MergeEntity_k.Args.entity_animation_layers_out, core_memory.get_buffer(CoreBufferType.ENTITY_ANIM_LAYER))
            .buf_arg(MergeEntity_k.Args.entity_previous_layers_out, core_memory.get_buffer(CoreBufferType.ENTITY_PREV_LAYER))
            .buf_arg(MergeEntity_k.Args.entity_hull_tables_out, core_memory.get_buffer(CoreBufferType.ENTITY_HULL_TABLE))
            .buf_arg(MergeEntity_k.Args.entity_bone_tables_out, core_memory.get_buffer(CoreBufferType.ENTITY_BONE_TABLE))
            .buf_arg(MergeEntity_k.Args.entity_masses_out, core_memory.get_buffer(CoreBufferType.ENTITY_MASS))
            .buf_arg(MergeEntity_k.Args.entity_root_hulls_out, core_memory.get_buffer(CoreBufferType.ENTITY_ROOT_HULL))
            .buf_arg(MergeEntity_k.Args.entity_model_indices_out, core_memory.get_buffer(CoreBufferType.ENTITY_MODEL_ID))
            .buf_arg(MergeEntity_k.Args.entity_model_transforms_out, core_memory.get_buffer(CoreBufferType.ENTITY_TRANSFORM_ID))
            .buf_arg(MergeEntity_k.Args.entity_types_out, core_memory.get_buffer(CoreBufferType.ENTITY_TYPE))
            .buf_arg(MergeEntity_k.Args.entity_flags_out, core_memory.get_buffer(CoreBufferType.ENTITY_FLAG));

        long k_ptr_merge_hull_bone = this.p_gpu_crud.kernel_ptr(KernelType.merge_hull_bone);
        k_merge_hull_bone = new MergeHullBone_k(ptr_queue, k_ptr_merge_hull_bone)
            .buf_arg(MergeHullBone_k.Args.hull_bones_in, buffers.buffer(CoreBufferType.HULL_BONE))
            .buf_arg(MergeHullBone_k.Args.hull_bind_pose_indicies_in, buffers.buffer(CoreBufferType.HULL_BONE_BIND_POSE))
            .buf_arg(MergeHullBone_k.Args.hull_inv_bind_pose_indicies_in, buffers.buffer(CoreBufferType.HULL_BONE_INV_BIND_POSE))
            .buf_arg(MergeHullBone_k.Args.hull_bones_out, core_memory.get_buffer(CoreBufferType.HULL_BONE))
            .buf_arg(MergeHullBone_k.Args.hull_bind_pose_indicies_out, core_memory.get_buffer(CoreBufferType.HULL_BONE_BIND_POSE))
            .buf_arg(MergeHullBone_k.Args.hull_inv_bind_pose_indicies_out, core_memory.get_buffer(CoreBufferType.HULL_BONE_INV_BIND_POSE));

        long k_ptr_merge_entity_bone = this.p_gpu_crud.kernel_ptr(KernelType.merge_entity_bone);
        k_merge_entity_bone = new MergeEntityBone_k(ptr_queue, k_ptr_merge_entity_bone)
            .buf_arg(MergeEntityBone_k.Args.armature_bones_in, buffers.buffer(CoreBufferType.ENTITY_BONE))
            .buf_arg(MergeEntityBone_k.Args.armature_bone_reference_ids_in, buffers.buffer(CoreBufferType.ENTITY_BONE_REFERENCE_ID))
            .buf_arg(MergeEntityBone_k.Args.armature_bone_parent_ids_in, buffers.buffer(CoreBufferType.ENTITY_BONE_PARENT_ID))
            .buf_arg(MergeEntityBone_k.Args.armature_bones_out, core_memory.get_buffer(CoreBufferType.ENTITY_BONE))
            .buf_arg(MergeEntityBone_k.Args.armature_bone_reference_ids_out, core_memory.get_buffer(CoreBufferType.ENTITY_BONE_REFERENCE_ID))
            .buf_arg(MergeEntityBone_k.Args.armature_bone_parent_ids_out, core_memory.get_buffer(CoreBufferType.ENTITY_BONE_PARENT_ID));
    }

    public void merge_into(SectorContainer target_container)
    {
        int point_count       = controller.next_point();
        int edge_count        = controller.next_edge();
        int hull_count        = controller.next_hull();
        int entity_count      = controller.next_entity();
        int hull_bone_count   = controller.next_hull_bone();
        int entity_bone_count = controller.next_entity_bone();

        int point_size       = GPGPU.calculate_preferred_global_size(point_count);
        int edge_size        = GPGPU.calculate_preferred_global_size(edge_count);
        int hull_size        = GPGPU.calculate_preferred_global_size(hull_count);
        int entity_size      = GPGPU.calculate_preferred_global_size(entity_count);
        int hull_bone_size   = GPGPU.calculate_preferred_global_size(hull_bone_count);
        int entity_bone_size = GPGPU.calculate_preferred_global_size(entity_bone_count);

        if (point_count > 0) k_merge_point
            .set_arg(MergePoint_k.Args.point_offset, target_container.next_point())
            .set_arg(MergePoint_k.Args.bone_offset,  target_container.next_hull_bone())
            .set_arg(MergePoint_k.Args.hull_offset,  target_container.next_hull())
            .set_arg(MergePoint_k.Args.max_point,    point_count)
            .call(arg_long(point_size), GPGPU.preferred_work_size);

        if (edge_count > 0) k_merge_edge
            .set_arg(MergeEdge_k.Args.edge_offset,  target_container.next_edge())
            .set_arg(MergeEdge_k.Args.point_offset, target_container.next_point())
            .set_arg(MergeEdge_k.Args.max_edge,     edge_count)
            .call(arg_long(edge_size), GPGPU.preferred_work_size);

        if (hull_count > 0) k_merge_hull
            .set_arg(MergeHull_k.Args.hull_offset,      target_container.next_hull())
            .set_arg(MergeHull_k.Args.point_offset,     target_container.next_point())
            .set_arg(MergeHull_k.Args.edge_offset,      target_container.next_edge())
            .set_arg(MergeHull_k.Args.entity_offset,    target_container.next_entity())
            .set_arg(MergeHull_k.Args.hull_bone_offset, target_container.next_hull_bone())
            .set_arg(MergeHull_k.Args.max_hull,         hull_count)
            .call(arg_long(hull_size), GPGPU.preferred_work_size);

        if (entity_count > 0) k_merge_entity
            .set_arg(MergeEntity_k.Args.entity_offset,        target_container.next_entity())
            .set_arg(MergeEntity_k.Args.hull_offset,          target_container.next_hull())
            .set_arg(MergeEntity_k.Args.armature_bone_offset, target_container.next_entity_bone())
            .set_arg(MergeEntity_k.Args.max_entity,           edge_count)
            .call(arg_long(entity_size), GPGPU.preferred_work_size);

        if (hull_bone_count > 0) k_merge_hull_bone
            .set_arg(MergeHullBone_k.Args.hull_bone_offset,     target_container.next_hull_bone())
            .set_arg(MergeHullBone_k.Args.armature_bone_offset, target_container.next_entity_bone())
            .set_arg(MergeHullBone_k.Args.max_hull_bone,        hull_bone_count)
            .call(arg_long(hull_bone_size), GPGPU.preferred_work_size);

        if (entity_bone_count > 0) k_merge_entity_bone
            .set_arg(MergeEntityBone_k.Args.armature_bone_offset, target_container.next_entity_bone())
            .set_arg(MergeEntityBone_k.Args.max_entity_bone,      entity_bone_count)
            .call(arg_long(entity_bone_size), GPGPU.preferred_work_size);

        controller.reset();
    }

    @Override
    public int next_point()
    {
        return controller.next_point();
    }

    @Override
    public int next_edge()
    {
        return controller.next_edge();
    }

    @Override
    public int next_hull()
    {
        return controller.next_hull();
    }

    @Override
    public int next_entity()
    {
        return controller.next_entity();
    }

    @Override
    public int next_hull_bone()
    {
        return controller.next_hull_bone();
    }

    @Override
    public int next_entity_bone()
    {
        return controller.next_entity_bone();
    }

    @Override
    public int create_point(float[] position, int[] bone_ids, int vertex_index, int hull_index, int hit_count, int flags)
    {
        return controller.create_point(position, bone_ids, vertex_index, hull_index, hit_count, flags);
    }

    @Override
    public int create_edge(int p1, int p2, float l, int flags, int edge_pin)
    {
        return controller.create_edge(p1, p2, l, flags, edge_pin);
    }

    @Override
    public int create_hull(int mesh_id, float[] position, float[] scale, float[] rotation, int[] point_table, int[] edge_table, int[] bone_table, float friction, float restitution, int entity_id, int uv_offset, int flags)
    {
        return controller.create_hull(mesh_id, position, scale, rotation, point_table, edge_table, bone_table, friction, restitution, entity_id, uv_offset, flags);
    }

    @Override
    public int create_entity(float x, float y, float z, float w, int[] hull_table, int[] bone_table, float mass, int anim_index, float anim_time, int root_hull, int model_id, int model_transform_id, int type, int flags)
    {
        return controller.create_entity(x, y, z, w, hull_table, bone_table, mass, anim_index, anim_time, root_hull, model_id, model_transform_id, type, flags);
    }

    @Override
    public int create_hull_bone(float[] bone_data, int bind_pose_id, int inv_bind_pose_id)
    {
        return controller.create_hull_bone(bone_data, bind_pose_id, inv_bind_pose_id);
    }

    @Override
    public int create_entity_bone(int bone_reference, int bone_parent_id, float[] bone_data)
    {
        return controller.create_entity_bone(bone_reference, bone_parent_id, bone_data);
    }

    @Override
    public void release()
    {
        p_gpu_crud.release();
        buffers.release();
        controller.release();
    }
}
