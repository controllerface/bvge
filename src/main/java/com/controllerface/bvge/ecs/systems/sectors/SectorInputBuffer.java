package com.controllerface.bvge.ecs.systems.sectors;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.buffers.BufferType;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.*;

public class SectorInputBuffer implements SectorContainer
{
    private static final long ENTITY_INIT = 1_000L;
    private static final long HULL_INIT = 1_000L;
    private static final long EDGE_INIT = 2_400L;
    private static final long POINT_INIT = 5_000L;

    private final GPUProgram p_gpu_crud;

    private final GPUKernel k_merge_point;
    private final GPUKernel k_merge_edge;
    private final GPUKernel k_merge_hull;
    private final GPUKernel k_merge_entity;
    private final GPUKernel k_merge_hull_bone;
    private final GPUKernel k_merge_entity_bone;

    private final long ptr_queue;
    private final OrderedSectorGroup sector_group;
    private final OrderedSectorInput sector_input;

    public SectorInputBuffer(long ptr_queue, GPUProgram p_gpu_crud, GPUCoreMemory core_memory)
    {
        this.ptr_queue  = ptr_queue;
        this.p_gpu_crud = p_gpu_crud;
        this.sector_group = new OrderedSectorGroup(this.ptr_queue, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        this.sector_input = new OrderedSectorInput(this.ptr_queue, this.p_gpu_crud, this.sector_group);

        long k_ptr_merge_point = this.p_gpu_crud.kernel_ptr(Kernel.merge_point);
        k_merge_point = new MergePoint_k(this.ptr_queue, k_ptr_merge_point)
            .buf_arg(MergePoint_k.Args.points_in, sector_group.buffer(BufferType.POINT))
            .buf_arg(MergePoint_k.Args.point_vertex_references_in, sector_group.buffer(BufferType.POINT_VERTEX_REFERENCE))
            .buf_arg(MergePoint_k.Args.point_hull_indices_in, sector_group.buffer(BufferType.POINT_HULL_INDEX))
            .buf_arg(MergePoint_k.Args.point_hit_counts_in, sector_group.buffer(BufferType.POINT_HIT_COUNT))
            .buf_arg(MergePoint_k.Args.point_flags_in, sector_group.buffer(BufferType.POINT_FLAG))
            .buf_arg(MergePoint_k.Args.point_bone_tables_in, sector_group.buffer(BufferType.POINT_BONE_TABLE))
            .buf_arg(MergePoint_k.Args.points_out, core_memory.buffer(BufferType.POINT))
            .buf_arg(MergePoint_k.Args.point_vertex_references_out, core_memory.buffer(BufferType.POINT_VERTEX_REFERENCE))
            .buf_arg(MergePoint_k.Args.point_hull_indices_out, core_memory.buffer(BufferType.POINT_HULL_INDEX))
            .buf_arg(MergePoint_k.Args.point_hit_counts_out, core_memory.buffer(BufferType.POINT_HIT_COUNT))
            .buf_arg(MergePoint_k.Args.point_flags_out, core_memory.buffer(BufferType.POINT_FLAG))
            .buf_arg(MergePoint_k.Args.point_bone_tables_out, core_memory.buffer(BufferType.POINT_BONE_TABLE));

        long k_ptr_merge_edge = this.p_gpu_crud.kernel_ptr(Kernel.merge_edge);
        k_merge_edge = new MergeEdge_k(this.ptr_queue, k_ptr_merge_edge)
            .buf_arg(MergeEdge_k.Args.edges_in, sector_group.buffer(BufferType.EDGE))
            .buf_arg(MergeEdge_k.Args.edge_lengths_in, sector_group.buffer(BufferType.EDGE_LENGTH))
            .buf_arg(MergeEdge_k.Args.edge_flags_in, sector_group.buffer(BufferType.EDGE_FLAG))
            .buf_arg(MergeEdge_k.Args.edges_out, core_memory.buffer(BufferType.EDGE))
            .buf_arg(MergeEdge_k.Args.edge_lengths_out, core_memory.buffer(BufferType.EDGE_LENGTH))
            .buf_arg(MergeEdge_k.Args.edge_flags_out, core_memory.buffer(BufferType.EDGE_FLAG));

        long k_ptr_merge_hull = this.p_gpu_crud.kernel_ptr(Kernel.merge_hull);
        k_merge_hull = new MergeHull_k(this.ptr_queue, k_ptr_merge_hull)
            .buf_arg(MergeHull_k.Args.hulls_in, sector_group.buffer(BufferType.HULL))
            .buf_arg(MergeHull_k.Args.hull_scales_in, sector_group.buffer(BufferType.HULL_SCALE))
            .buf_arg(MergeHull_k.Args.hull_rotations_in, sector_group.buffer(BufferType.HULL_ROTATION))
            .buf_arg(MergeHull_k.Args.hull_frictions_in, sector_group.buffer(BufferType.HULL_FRICTION))
            .buf_arg(MergeHull_k.Args.hull_restitutions_in, sector_group.buffer(BufferType.HULL_RESTITUTION))
            .buf_arg(MergeHull_k.Args.hull_point_tables_in, sector_group.buffer(BufferType.HULL_POINT_TABLE))
            .buf_arg(MergeHull_k.Args.hull_edge_tables_in, sector_group.buffer(BufferType.HULL_EDGE_TABLE))
            .buf_arg(MergeHull_k.Args.hull_bone_tables_in, sector_group.buffer(BufferType.HULL_BONE_TABLE))
            .buf_arg(MergeHull_k.Args.hull_entity_ids_in, sector_group.buffer(BufferType.HULL_ENTITY_ID))
            .buf_arg(MergeHull_k.Args.hull_flags_in, sector_group.buffer(BufferType.HULL_FLAG))
            .buf_arg(MergeHull_k.Args.hull_mesh_ids_in, sector_group.buffer(BufferType.HULL_MESH_ID))
            .buf_arg(MergeHull_k.Args.hull_uv_offsets_in, sector_group.buffer(BufferType.HULL_UV_OFFSET))
            .buf_arg(MergeHull_k.Args.hull_integrity_in, sector_group.buffer(BufferType.HULL_INTEGRITY))
            .buf_arg(MergeHull_k.Args.hulls_out, core_memory.buffer(BufferType.HULL))
            .buf_arg(MergeHull_k.Args.hull_scales_out, core_memory.buffer(BufferType.HULL_SCALE))
            .buf_arg(MergeHull_k.Args.hull_rotations_out, core_memory.buffer(BufferType.HULL_ROTATION))
            .buf_arg(MergeHull_k.Args.hull_frictions_out, core_memory.buffer(BufferType.HULL_FRICTION))
            .buf_arg(MergeHull_k.Args.hull_restitutions_out, core_memory.buffer(BufferType.HULL_RESTITUTION))
            .buf_arg(MergeHull_k.Args.hull_point_tables_out, core_memory.buffer(BufferType.HULL_POINT_TABLE))
            .buf_arg(MergeHull_k.Args.hull_edge_tables_out, core_memory.buffer(BufferType.HULL_EDGE_TABLE))
            .buf_arg(MergeHull_k.Args.hull_bone_tables_out, core_memory.buffer(BufferType.HULL_BONE_TABLE))
            .buf_arg(MergeHull_k.Args.hull_entity_ids_out, core_memory.buffer(BufferType.HULL_ENTITY_ID))
            .buf_arg(MergeHull_k.Args.hull_flags_out, core_memory.buffer(BufferType.HULL_FLAG))
            .buf_arg(MergeHull_k.Args.hull_mesh_ids_out, core_memory.buffer(BufferType.HULL_MESH_ID))
            .buf_arg(MergeHull_k.Args.hull_uv_offsets_out, core_memory.buffer(BufferType.HULL_UV_OFFSET))
            .buf_arg(MergeHull_k.Args.hull_integrity_out, core_memory.buffer(BufferType.HULL_INTEGRITY));

        long k_ptr_merge_entity = this.p_gpu_crud.kernel_ptr(Kernel.merge_entity);
        k_merge_entity = new MergeEntity_k(this.ptr_queue, k_ptr_merge_entity)
            .buf_arg(MergeEntity_k.Args.entities_in, sector_group.buffer(BufferType.ENTITY))
            .buf_arg(MergeEntity_k.Args.entity_animation_elapsed_in, sector_group.buffer(BufferType.ENTITY_ANIM_ELAPSED))
            .buf_arg(MergeEntity_k.Args.entity_motion_states_in, sector_group.buffer(BufferType.ENTITY_MOTION_STATE))
            .buf_arg(MergeEntity_k.Args.entity_animation_indices_in, sector_group.buffer(BufferType.ENTITY_ANIM_INDEX))
            .buf_arg(MergeEntity_k.Args.entity_hull_tables_in, sector_group.buffer(BufferType.ENTITY_HULL_TABLE))
            .buf_arg(MergeEntity_k.Args.entity_bone_tables_in, sector_group.buffer(BufferType.ENTITY_BONE_TABLE))
            .buf_arg(MergeEntity_k.Args.entity_masses_in, sector_group.buffer(BufferType.ENTITY_MASS))
            .buf_arg(MergeEntity_k.Args.entity_root_hulls_in, sector_group.buffer(BufferType.ENTITY_ROOT_HULL))
            .buf_arg(MergeEntity_k.Args.entity_model_indices_in, sector_group.buffer(BufferType.ENTITY_MODEL_ID))
            .buf_arg(MergeEntity_k.Args.entity_model_transforms_in, sector_group.buffer(BufferType.ENTITY_TRANSFORM_ID))
            .buf_arg(MergeEntity_k.Args.entity_flags_in, sector_group.buffer(BufferType.ENTITY_FLAG))
            .buf_arg(MergeEntity_k.Args.entities_out, core_memory.buffer(BufferType.ENTITY))
            .buf_arg(MergeEntity_k.Args.entity_animation_elapsed_out, core_memory.buffer(BufferType.ENTITY_ANIM_ELAPSED))
            .buf_arg(MergeEntity_k.Args.entity_motion_states_out, core_memory.buffer(BufferType.ENTITY_MOTION_STATE))
            .buf_arg(MergeEntity_k.Args.entity_animation_indices_out, core_memory.buffer(BufferType.ENTITY_ANIM_INDEX))
            .buf_arg(MergeEntity_k.Args.entity_hull_tables_out, core_memory.buffer(BufferType.ENTITY_HULL_TABLE))
            .buf_arg(MergeEntity_k.Args.entity_bone_tables_out, core_memory.buffer(BufferType.ENTITY_BONE_TABLE))
            .buf_arg(MergeEntity_k.Args.entity_masses_out, core_memory.buffer(BufferType.ENTITY_MASS))
            .buf_arg(MergeEntity_k.Args.entity_root_hulls_out, core_memory.buffer(BufferType.ENTITY_ROOT_HULL))
            .buf_arg(MergeEntity_k.Args.entity_model_indices_out, core_memory.buffer(BufferType.ENTITY_MODEL_ID))
            .buf_arg(MergeEntity_k.Args.entity_model_transforms_out, core_memory.buffer(BufferType.ENTITY_TRANSFORM_ID))
            .buf_arg(MergeEntity_k.Args.entity_flags_out, core_memory.buffer(BufferType.ENTITY_FLAG));

        long k_ptr_merge_hull_bone = this.p_gpu_crud.kernel_ptr(Kernel.merge_hull_bone);
        k_merge_hull_bone = new MergeHullBone_k(this.ptr_queue, k_ptr_merge_hull_bone)
            .buf_arg(MergeHullBone_k.Args.hull_bones_in, sector_group.buffer(BufferType.HULL_BONE))
            .buf_arg(MergeHullBone_k.Args.hull_bind_pose_indicies_in, sector_group.buffer(BufferType.HULL_BONE_BIND_POSE))
            .buf_arg(MergeHullBone_k.Args.hull_inv_bind_pose_indicies_in, sector_group.buffer(BufferType.HULL_BONE_INV_BIND_POSE))
            .buf_arg(MergeHullBone_k.Args.hull_bones_out, core_memory.buffer(BufferType.HULL_BONE))
            .buf_arg(MergeHullBone_k.Args.hull_bind_pose_indicies_out, core_memory.buffer(BufferType.HULL_BONE_BIND_POSE))
            .buf_arg(MergeHullBone_k.Args.hull_inv_bind_pose_indicies_out, core_memory.buffer(BufferType.HULL_BONE_INV_BIND_POSE));

        long k_ptr_merge_entity_bone = this.p_gpu_crud.kernel_ptr(Kernel.merge_entity_bone);
        k_merge_entity_bone = new MergeEntityBone_k(this.ptr_queue, k_ptr_merge_entity_bone)
            .buf_arg(MergeEntityBone_k.Args.armature_bones_in, sector_group.buffer(BufferType.ENTITY_BONE))
            .buf_arg(MergeEntityBone_k.Args.armature_bone_reference_ids_in, sector_group.buffer(BufferType.ENTITY_BONE_REFERENCE_ID))
            .buf_arg(MergeEntityBone_k.Args.armature_bone_parent_ids_in, sector_group.buffer(BufferType.ENTITY_BONE_PARENT_ID))
            .buf_arg(MergeEntityBone_k.Args.armature_bones_out, core_memory.buffer(BufferType.ENTITY_BONE))
            .buf_arg(MergeEntityBone_k.Args.armature_bone_reference_ids_out, core_memory.buffer(BufferType.ENTITY_BONE_REFERENCE_ID))
            .buf_arg(MergeEntityBone_k.Args.armature_bone_parent_ids_out, core_memory.buffer(BufferType.ENTITY_BONE_PARENT_ID));
    }

    public SectorInputBuffer(long ptr_queue, GPUCoreMemory core_memory)
    {
        this(ptr_queue, new GPUCrud().init(), core_memory);
    }

    public void merge_into_parent(SectorContainer parent)
    {
        if (sector_input.point_index() > 0) k_merge_point
            .set_arg(MergePoint_k.Args.point_offset, parent.next_point())
            .set_arg(MergePoint_k.Args.bone_offset,  parent.next_hull_bone())
            .set_arg(MergePoint_k.Args.hull_offset,  parent.next_hull())
            .call(arg_long(sector_input.point_index()));

        if (sector_input.edge_index() > 0) k_merge_edge
            .set_arg(MergeEdge_k.Args.edge_offset,  parent.next_edge())
            .set_arg(MergeEdge_k.Args.point_offset, parent.next_point())
            .call(arg_long(sector_input.edge_index()));

        if (sector_input.hull_index() > 0) k_merge_hull
            .set_arg(MergeHull_k.Args.hull_offset,      parent.next_hull())
            .set_arg(MergeHull_k.Args.point_offset,     parent.next_point())
            .set_arg(MergeHull_k.Args.edge_offset,      parent.next_edge())
            .set_arg(MergeHull_k.Args.entity_offset,    parent.next_entity())
            .set_arg(MergeHull_k.Args.hull_bone_offset, parent.next_hull_bone())
            .call(arg_long(sector_input.hull_index()));

        if (sector_input.entity_index() > 0) k_merge_entity
            .set_arg(MergeEntity_k.Args.entity_offset,        parent.next_entity())
            .set_arg(MergeEntity_k.Args.hull_offset,          parent.next_hull())
            .set_arg(MergeEntity_k.Args.armature_bone_offset, parent.next_armature_bone())
            .call(arg_long(sector_input.entity_index()));

        if (sector_input.hull_bone_index() > 0) k_merge_hull_bone
            .set_arg(MergeHullBone_k.Args.hull_bone_offset,     parent.next_hull_bone())
            .set_arg(MergeHullBone_k.Args.armature_bone_offset, parent.next_armature_bone())
            .call(arg_long(sector_input.hull_bone_index()));

        if (sector_input.entity_bone_index() > 0) k_merge_entity_bone
            .set_arg(MergeEntityBone_k.Args.armature_bone_offset, parent.next_armature_bone())
            .call(arg_long(sector_input.entity_bone_index()));

        sector_input.reset();
    }

    @Override
    public int next_point()
    {
        return sector_input.point_index();
    }

    @Override
    public int next_edge()
    {
        return sector_input.edge_index();
    }

    @Override
    public int next_hull()
    {
        return sector_input.hull_index();
    }

    @Override
    public int next_entity()
    {
        return sector_input.entity_index();
    }

    @Override
    public int next_hull_bone()
    {
        return sector_input.hull_bone_index();
    }

    @Override
    public int next_armature_bone()
    {
        return sector_input.entity_bone_index();
    }

    @Override
    public int new_point(float[] position, int[] bone_ids, int vertex_index, int hull_index, int hit_count, int flags)
    {
        return sector_input.create_point(position, bone_ids, vertex_index, hull_index, hit_count, flags);
    }

    @Override
    public int new_edge(int p1, int p2, float l, int flags)
    {
        return sector_input.create_edge(p1, p2, l, flags);
    }

    @Override
    public int new_hull(int mesh_id, float[] position, float[] scale, float[] rotation, int[] point_table, int[] edge_table, int[] bone_table, float friction, float restitution, int entity_id, int uv_offset, int flags)
    {
        return sector_input.create_hull(mesh_id, position, scale, rotation, point_table, edge_table, bone_table, friction, restitution, entity_id, uv_offset, flags);
    }

    @Override
    public int new_entity(float x, float y, float z, float w, int[] hull_table, int[] bone_table, float mass, int anim_index, float anim_time, int root_hull, int model_id, int model_transform_id, int flags)
    {
        return sector_input.create_entity(x, y, z, w, hull_table, bone_table, mass, anim_index, anim_time, root_hull, model_id, model_transform_id, flags);
    }

    @Override
    public int new_hull_bone(float[] bone_data, int bind_pose_id, int inv_bind_pose_id)
    {
        return sector_input.create_hull_bone(bone_data, bind_pose_id, inv_bind_pose_id);
    }

    @Override
    public int new_armature_bone(int bone_reference, int bone_parent_id, float[] bone_data)
    {
        return sector_input.create_armature_bone(bone_reference, bone_parent_id, bone_data);
    }

    @Override
    public void destroy()
    {
        p_gpu_crud.destroy();
        sector_group.destroy();
    }
}
