package com.controllerface.bvge.ecs.systems.sectors;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.buffers.BufferType;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.*;

public class SectorOutputBuffer
{
    private static final long ENTITY_INIT = 1_000L;
    private static final long HULL_INIT = 1_000L;
    private static final long EDGE_INIT = 2_400L;
    private static final long POINT_INIT = 5_000L;

    private final GPUProgram p_gpu_crud = new GPUCrud();
    private final GPUKernel k_egress_entities;
    private final long ptr_queue;
    private final long ptr_egress_sizes;
    private final UnorderedSectorGroup sector_group;

    public SectorOutputBuffer(long ptr_queue, GPUCoreMemory core_memory)
    {
        this.ptr_queue         = ptr_queue;
        this.ptr_egress_sizes  = GPGPU.cl_new_pinned_buffer(CLSize.cl_int * 6);
        this.sector_group = new UnorderedSectorGroup(this.ptr_queue, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        p_gpu_crud.init();

        long k_ptr_egress_candidates = p_gpu_crud.kernel_ptr(Kernel.egress_entities);
        k_egress_entities = new EgressEntities_k(GPGPU.ptr_compute_queue, k_ptr_egress_candidates)
            .buf_arg(EgressEntities_k.Args.points_in,                       core_memory.buffer(BufferType.POINT))
            .buf_arg(EgressEntities_k.Args.point_vertex_references_in,      core_memory.buffer(BufferType.POINT_VERTEX_REFERENCE))
            .buf_arg(EgressEntities_k.Args.point_hull_indices_in,           core_memory.buffer(BufferType.POINT_HULL_INDEX))
            .buf_arg(EgressEntities_k.Args.point_hit_counts_in,             core_memory.buffer(BufferType.POINT_HIT_COUNT))
            .buf_arg(EgressEntities_k.Args.point_flags_in,                  core_memory.buffer(BufferType.POINT_FLAG))
            .buf_arg(EgressEntities_k.Args.point_bone_tables_in,            core_memory.buffer(BufferType.POINT_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.edges_in,                        core_memory.buffer(BufferType.EDGE))
            .buf_arg(EgressEntities_k.Args.edge_lengths_in,                 core_memory.buffer(BufferType.EDGE_LENGTH))
            .buf_arg(EgressEntities_k.Args.edge_flags_in,                   core_memory.buffer(BufferType.EDGE_FLAG))
            .buf_arg(EgressEntities_k.Args.hulls_in,                        core_memory.buffer(BufferType.HULL))
            .buf_arg(EgressEntities_k.Args.hull_scales_in,                  core_memory.buffer(BufferType.HULL_SCALE))
            .buf_arg(EgressEntities_k.Args.hull_rotations_in,               core_memory.buffer(BufferType.HULL_ROTATION))
            .buf_arg(EgressEntities_k.Args.hull_frictions_in,               core_memory.buffer(BufferType.HULL_FRICTION))
            .buf_arg(EgressEntities_k.Args.hull_restitutions_in,            core_memory.buffer(BufferType.HULL_RESTITUTION))
            .buf_arg(EgressEntities_k.Args.hull_point_tables_in,            core_memory.buffer(BufferType.HULL_POINT_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_edge_tables_in,             core_memory.buffer(BufferType.HULL_EDGE_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_bone_tables_in,             core_memory.buffer(BufferType.HULL_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_entity_ids_in,              core_memory.buffer(BufferType.HULL_ENTITY_ID))
            .buf_arg(EgressEntities_k.Args.hull_flags_in,                   core_memory.buffer(BufferType.HULL_FLAG))
            .buf_arg(EgressEntities_k.Args.hull_mesh_ids_in,                core_memory.buffer(BufferType.HULL_MESH_ID))
            .buf_arg(EgressEntities_k.Args.hull_uv_offsets_in,              core_memory.buffer(BufferType.HULL_UV_OFFSET))
            .buf_arg(EgressEntities_k.Args.hull_integrity_in,               core_memory.buffer(BufferType.HULL_INTEGRITY))
            .buf_arg(EgressEntities_k.Args.entities_in,                     core_memory.buffer(BufferType.ENTITY))
            .buf_arg(EgressEntities_k.Args.entity_animation_elapsed_in,     core_memory.buffer(BufferType.ENTITY_ANIM_ELAPSED))
            .buf_arg(EgressEntities_k.Args.entity_motion_states_in,         core_memory.buffer(BufferType.ENTITY_MOTION_STATE))
            .buf_arg(EgressEntities_k.Args.entity_animation_indices_in,     core_memory.buffer(BufferType.ENTITY_ANIM_INDEX))
            .buf_arg(EgressEntities_k.Args.entity_hull_tables_in,           core_memory.buffer(BufferType.ENTITY_HULL_TABLE))
            .buf_arg(EgressEntities_k.Args.entity_bone_tables_in,           core_memory.buffer(BufferType.ENTITY_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.entity_masses_in,                core_memory.buffer(BufferType.ENTITY_MASS))
            .buf_arg(EgressEntities_k.Args.entity_root_hulls_in,            core_memory.buffer(BufferType.ENTITY_ROOT_HULL))
            .buf_arg(EgressEntities_k.Args.entity_model_indices_in,         core_memory.buffer(BufferType.ENTITY_MODEL_ID))
            .buf_arg(EgressEntities_k.Args.entity_model_transforms_in,      core_memory.buffer(BufferType.ENTITY_TRANSFORM_ID))
            .buf_arg(EgressEntities_k.Args.entity_flags_in,                 core_memory.buffer(BufferType.ENTITY_FLAG))
            .buf_arg(EgressEntities_k.Args.hull_bones_in,                   core_memory.buffer(BufferType.HULL_BONE))
            .buf_arg(EgressEntities_k.Args.hull_bind_pose_indicies_in,      core_memory.buffer(BufferType.HULL_BONE_BIND_POSE))
            .buf_arg(EgressEntities_k.Args.hull_inv_bind_pose_indicies_in,  core_memory.buffer(BufferType.HULL_BONE_INV_BIND_POSE))
            .buf_arg(EgressEntities_k.Args.armature_bones_in,               core_memory.buffer(BufferType.ENTITY_BONE))
            .buf_arg(EgressEntities_k.Args.armature_bone_reference_ids_in,  core_memory.buffer(BufferType.ENTITY_BONE_REFERENCE_ID))
            .buf_arg(EgressEntities_k.Args.armature_bone_parent_ids_in,     core_memory.buffer(BufferType.ENTITY_BONE_PARENT_ID))
            .buf_arg(EgressEntities_k.Args.points_out,                      sector_group.buffer(BufferType.POINT))
            .buf_arg(EgressEntities_k.Args.point_vertex_references_out,     sector_group.buffer(BufferType.POINT_VERTEX_REFERENCE))
            .buf_arg(EgressEntities_k.Args.point_hull_indices_out,          sector_group.buffer(BufferType.POINT_HULL_INDEX))
            .buf_arg(EgressEntities_k.Args.point_hit_counts_out,            sector_group.buffer(BufferType.POINT_HIT_COUNT))
            .buf_arg(EgressEntities_k.Args.point_flags_out,                 sector_group.buffer(BufferType.POINT_FLAG))
            .buf_arg(EgressEntities_k.Args.point_bone_tables_out,           sector_group.buffer(BufferType.POINT_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.edges_out,                       sector_group.buffer(BufferType.EDGE))
            .buf_arg(EgressEntities_k.Args.edge_lengths_out,                sector_group.buffer(BufferType.EDGE_LENGTH))
            .buf_arg(EgressEntities_k.Args.edge_flags_out,                  sector_group.buffer(BufferType.EDGE_FLAG))
            .buf_arg(EgressEntities_k.Args.hulls_out,                       sector_group.buffer(BufferType.HULL))
            .buf_arg(EgressEntities_k.Args.hull_scales_out,                 sector_group.buffer(BufferType.HULL_SCALE))
            .buf_arg(EgressEntities_k.Args.hull_rotations_out,              sector_group.buffer(BufferType.HULL_ROTATION))
            .buf_arg(EgressEntities_k.Args.hull_frictions_out,              sector_group.buffer(BufferType.HULL_FRICTION))
            .buf_arg(EgressEntities_k.Args.hull_restitutions_out,           sector_group.buffer(BufferType.HULL_RESTITUTION))
            .buf_arg(EgressEntities_k.Args.hull_point_tables_out,           sector_group.buffer(BufferType.HULL_POINT_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_edge_tables_out,            sector_group.buffer(BufferType.HULL_EDGE_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_bone_tables_out,            sector_group.buffer(BufferType.HULL_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_entity_ids_out,             sector_group.buffer(BufferType.HULL_ENTITY_ID))
            .buf_arg(EgressEntities_k.Args.hull_flags_out,                  sector_group.buffer(BufferType.HULL_FLAG))
            .buf_arg(EgressEntities_k.Args.hull_mesh_ids_out,               sector_group.buffer(BufferType.HULL_MESH_ID))
            .buf_arg(EgressEntities_k.Args.hull_uv_offsets_out,             sector_group.buffer(BufferType.HULL_UV_OFFSET))
            .buf_arg(EgressEntities_k.Args.hull_integrity_out,              sector_group.buffer(BufferType.HULL_INTEGRITY))
            .buf_arg(EgressEntities_k.Args.entities_out,                    sector_group.buffer(BufferType.ENTITY))
            .buf_arg(EgressEntities_k.Args.entity_animation_elapsed_out,    sector_group.buffer(BufferType.ENTITY_ANIM_ELAPSED))
            .buf_arg(EgressEntities_k.Args.entity_motion_states_out,        sector_group.buffer(BufferType.ENTITY_MOTION_STATE))
            .buf_arg(EgressEntities_k.Args.entity_animation_indices_out,    sector_group.buffer(BufferType.ENTITY_ANIM_INDEX))
            .buf_arg(EgressEntities_k.Args.entity_hull_tables_out,          sector_group.buffer(BufferType.ENTITY_HULL_TABLE))
            .buf_arg(EgressEntities_k.Args.entity_bone_tables_out,          sector_group.buffer(BufferType.ENTITY_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.entity_masses_out,               sector_group.buffer(BufferType.ENTITY_MASS))
            .buf_arg(EgressEntities_k.Args.entity_root_hulls_out,           sector_group.buffer(BufferType.ENTITY_ROOT_HULL))
            .buf_arg(EgressEntities_k.Args.entity_model_indices_out,        sector_group.buffer(BufferType.ENTITY_MODEL_ID))
            .buf_arg(EgressEntities_k.Args.entity_model_transforms_out,     sector_group.buffer(BufferType.ENTITY_TRANSFORM_ID))
            .buf_arg(EgressEntities_k.Args.entity_flags_out,                sector_group.buffer(BufferType.ENTITY_FLAG))
            .buf_arg(EgressEntities_k.Args.hull_bones_out,                  sector_group.buffer(BufferType.HULL_BONE))
            .buf_arg(EgressEntities_k.Args.hull_bind_pose_indicies_out,     sector_group.buffer(BufferType.HULL_BONE_BIND_POSE))
            .buf_arg(EgressEntities_k.Args.hull_inv_bind_pose_indicies_out, sector_group.buffer(BufferType.HULL_BONE_INV_BIND_POSE))
            .buf_arg(EgressEntities_k.Args.armature_bones_out,              sector_group.buffer(BufferType.ENTITY_BONE))
            .buf_arg(EgressEntities_k.Args.armature_bone_reference_ids_out, sector_group.buffer(BufferType.ENTITY_BONE_REFERENCE_ID))
            .buf_arg(EgressEntities_k.Args.armature_bone_parent_ids_out,    sector_group.buffer(BufferType.ENTITY_BONE_PARENT_ID))
            .ptr_arg(EgressEntities_k.Args.counters,                        ptr_egress_sizes);
    }

    public void pull_from_parent(int entity_count, int[] egress_counts)
    {
        GPGPU.cl_zero_buffer(ptr_queue, ptr_egress_sizes, CLSize.cl_int * 6);
        sector_group.ensure_capacity(egress_counts);
        k_egress_entities.call(arg_long(entity_count));
    }

    public void unload_sector(UnorderedSectorGroup.Raw raw_sectors, int[] counts)
    {
        sector_group.unload_sectors(raw_sectors, counts);
    }

    public void destroy()
    {
        p_gpu_crud.destroy();
        sector_group.destroy();
    }
}
