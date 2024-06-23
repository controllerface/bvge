package com.controllerface.bvge.game.world.sectors;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.GPUProgram;

import static com.controllerface.bvge.cl.CLSize.*;
import static com.controllerface.bvge.cl.CLUtils.*;
import static com.controllerface.bvge.cl.buffers.CoreBufferType.*;

public class UnorderedSectorOutput
{
    private static final long ENTITY_INIT = 1_000L;
    private static final long HULL_INIT   = 1_000L;
    private static final long EDGE_INIT   = 2_400L;
    private static final long POINT_INIT  = 5_000L;

    private final GPUProgram p_gpu_crud;
    private final GPUKernel k_egress_entities;
    private final long ptr_queue;
    private final long ptr_egress_sizes;
    private final UnorderedSectorBufferGroup sector_buffers;

    public UnorderedSectorOutput(String name, long ptr_queue, GPUCoreMemory core_memory)
    {
        this.ptr_queue         = ptr_queue;
        this.ptr_egress_sizes  = GPGPU.cl_new_pinned_buffer(cl_int * 6);
        this.sector_buffers = new UnorderedSectorBufferGroup(name, this.ptr_queue, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        this.p_gpu_crud = new GPUCrud().init();

        long k_ptr_egress_candidates = p_gpu_crud.kernel_ptr(Kernel.egress_entities);
        k_egress_entities = new EgressEntities_k(this.ptr_queue, k_ptr_egress_candidates)
            .buf_arg(EgressEntities_k.Args.points_in,                       core_memory.get_buffer(POINT))
            .buf_arg(EgressEntities_k.Args.point_vertex_references_in,      core_memory.get_buffer(POINT_VERTEX_REFERENCE))
            .buf_arg(EgressEntities_k.Args.point_hull_indices_in,           core_memory.get_buffer(POINT_HULL_INDEX))
            .buf_arg(EgressEntities_k.Args.point_hit_counts_in,             core_memory.get_buffer(POINT_HIT_COUNT))
            .buf_arg(EgressEntities_k.Args.point_flags_in,                  core_memory.get_buffer(POINT_FLAG))
            .buf_arg(EgressEntities_k.Args.point_bone_tables_in,            core_memory.get_buffer(POINT_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.edges_in,                        core_memory.get_buffer(EDGE))
            .buf_arg(EgressEntities_k.Args.edge_lengths_in,                 core_memory.get_buffer(EDGE_LENGTH))
            .buf_arg(EgressEntities_k.Args.edge_flags_in,                   core_memory.get_buffer(EDGE_FLAG))
            .buf_arg(EgressEntities_k.Args.hulls_in,                        core_memory.get_buffer(HULL))
            .buf_arg(EgressEntities_k.Args.hull_scales_in,                  core_memory.get_buffer(HULL_SCALE))
            .buf_arg(EgressEntities_k.Args.hull_rotations_in,               core_memory.get_buffer(HULL_ROTATION))
            .buf_arg(EgressEntities_k.Args.hull_frictions_in,               core_memory.get_buffer(HULL_FRICTION))
            .buf_arg(EgressEntities_k.Args.hull_restitutions_in,            core_memory.get_buffer(HULL_RESTITUTION))
            .buf_arg(EgressEntities_k.Args.hull_point_tables_in,            core_memory.get_buffer(HULL_POINT_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_edge_tables_in,             core_memory.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_bone_tables_in,             core_memory.get_buffer(HULL_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_entity_ids_in,              core_memory.get_buffer(HULL_ENTITY_ID))
            .buf_arg(EgressEntities_k.Args.hull_flags_in,                   core_memory.get_buffer(HULL_FLAG))
            .buf_arg(EgressEntities_k.Args.hull_mesh_ids_in,                core_memory.get_buffer(HULL_MESH_ID))
            .buf_arg(EgressEntities_k.Args.hull_uv_offsets_in,              core_memory.get_buffer(HULL_UV_OFFSET))
            .buf_arg(EgressEntities_k.Args.hull_integrity_in,               core_memory.get_buffer(HULL_INTEGRITY))
            .buf_arg(EgressEntities_k.Args.entities_in,                     core_memory.get_buffer(ENTITY))
            .buf_arg(EgressEntities_k.Args.entity_animation_elapsed_in,     core_memory.get_buffer(ENTITY_ANIM_ELAPSED))
            .buf_arg(EgressEntities_k.Args.entity_motion_states_in,         core_memory.get_buffer(ENTITY_MOTION_STATE))
            .buf_arg(EgressEntities_k.Args.entity_animation_indices_in,     core_memory.get_buffer(ENTITY_ANIM_INDEX))
            .buf_arg(EgressEntities_k.Args.entity_hull_tables_in,           core_memory.get_buffer(ENTITY_HULL_TABLE))
            .buf_arg(EgressEntities_k.Args.entity_bone_tables_in,           core_memory.get_buffer(ENTITY_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.entity_masses_in,                core_memory.get_buffer(ENTITY_MASS))
            .buf_arg(EgressEntities_k.Args.entity_root_hulls_in,            core_memory.get_buffer(ENTITY_ROOT_HULL))
            .buf_arg(EgressEntities_k.Args.entity_model_indices_in,         core_memory.get_buffer(ENTITY_MODEL_ID))
            .buf_arg(EgressEntities_k.Args.entity_model_transforms_in,      core_memory.get_buffer(ENTITY_TRANSFORM_ID))
            .buf_arg(EgressEntities_k.Args.entity_types_in,                 core_memory.get_buffer(ENTITY_TYPE))
            .buf_arg(EgressEntities_k.Args.entity_flags_in,                 core_memory.get_buffer(ENTITY_FLAG))
            .buf_arg(EgressEntities_k.Args.hull_bones_in,                   core_memory.get_buffer(HULL_BONE))
            .buf_arg(EgressEntities_k.Args.hull_bind_pose_indicies_in,      core_memory.get_buffer(HULL_BONE_BIND_POSE))
            .buf_arg(EgressEntities_k.Args.hull_inv_bind_pose_indicies_in,  core_memory.get_buffer(HULL_BONE_INV_BIND_POSE))
            .buf_arg(EgressEntities_k.Args.armature_bones_in,               core_memory.get_buffer(ENTITY_BONE))
            .buf_arg(EgressEntities_k.Args.armature_bone_reference_ids_in,  core_memory.get_buffer(ENTITY_BONE_REFERENCE_ID))
            .buf_arg(EgressEntities_k.Args.armature_bone_parent_ids_in,     core_memory.get_buffer(ENTITY_BONE_PARENT_ID))
            .buf_arg(EgressEntities_k.Args.points_out,                      sector_buffers.get_buffer(POINT))
            .buf_arg(EgressEntities_k.Args.point_vertex_references_out,     sector_buffers.get_buffer(POINT_VERTEX_REFERENCE))
            .buf_arg(EgressEntities_k.Args.point_hull_indices_out,          sector_buffers.get_buffer(POINT_HULL_INDEX))
            .buf_arg(EgressEntities_k.Args.point_hit_counts_out,            sector_buffers.get_buffer(POINT_HIT_COUNT))
            .buf_arg(EgressEntities_k.Args.point_flags_out,                 sector_buffers.get_buffer(POINT_FLAG))
            .buf_arg(EgressEntities_k.Args.point_bone_tables_out,           sector_buffers.get_buffer(POINT_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.edges_out,                       sector_buffers.get_buffer(EDGE))
            .buf_arg(EgressEntities_k.Args.edge_lengths_out,                sector_buffers.get_buffer(EDGE_LENGTH))
            .buf_arg(EgressEntities_k.Args.edge_flags_out,                  sector_buffers.get_buffer(EDGE_FLAG))
            .buf_arg(EgressEntities_k.Args.hulls_out,                       sector_buffers.get_buffer(HULL))
            .buf_arg(EgressEntities_k.Args.hull_scales_out,                 sector_buffers.get_buffer(HULL_SCALE))
            .buf_arg(EgressEntities_k.Args.hull_rotations_out,              sector_buffers.get_buffer(HULL_ROTATION))
            .buf_arg(EgressEntities_k.Args.hull_frictions_out,              sector_buffers.get_buffer(HULL_FRICTION))
            .buf_arg(EgressEntities_k.Args.hull_restitutions_out,           sector_buffers.get_buffer(HULL_RESTITUTION))
            .buf_arg(EgressEntities_k.Args.hull_point_tables_out,           sector_buffers.get_buffer(HULL_POINT_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_edge_tables_out,            sector_buffers.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_bone_tables_out,            sector_buffers.get_buffer(HULL_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_entity_ids_out,             sector_buffers.get_buffer(HULL_ENTITY_ID))
            .buf_arg(EgressEntities_k.Args.hull_flags_out,                  sector_buffers.get_buffer(HULL_FLAG))
            .buf_arg(EgressEntities_k.Args.hull_mesh_ids_out,               sector_buffers.get_buffer(HULL_MESH_ID))
            .buf_arg(EgressEntities_k.Args.hull_uv_offsets_out,             sector_buffers.get_buffer(HULL_UV_OFFSET))
            .buf_arg(EgressEntities_k.Args.hull_integrity_out,              sector_buffers.get_buffer(HULL_INTEGRITY))
            .buf_arg(EgressEntities_k.Args.entities_out,                    sector_buffers.get_buffer(ENTITY))
            .buf_arg(EgressEntities_k.Args.entity_animation_elapsed_out,    sector_buffers.get_buffer(ENTITY_ANIM_ELAPSED))
            .buf_arg(EgressEntities_k.Args.entity_motion_states_out,        sector_buffers.get_buffer(ENTITY_MOTION_STATE))
            .buf_arg(EgressEntities_k.Args.entity_animation_indices_out,    sector_buffers.get_buffer(ENTITY_ANIM_INDEX))
            .buf_arg(EgressEntities_k.Args.entity_hull_tables_out,          sector_buffers.get_buffer(ENTITY_HULL_TABLE))
            .buf_arg(EgressEntities_k.Args.entity_bone_tables_out,          sector_buffers.get_buffer(ENTITY_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.entity_masses_out,               sector_buffers.get_buffer(ENTITY_MASS))
            .buf_arg(EgressEntities_k.Args.entity_root_hulls_out,           sector_buffers.get_buffer(ENTITY_ROOT_HULL))
            .buf_arg(EgressEntities_k.Args.entity_model_indices_out,        sector_buffers.get_buffer(ENTITY_MODEL_ID))
            .buf_arg(EgressEntities_k.Args.entity_model_transforms_out,     sector_buffers.get_buffer(ENTITY_TRANSFORM_ID))
            .buf_arg(EgressEntities_k.Args.entity_types_out,                sector_buffers.get_buffer(ENTITY_TYPE))
            .buf_arg(EgressEntities_k.Args.entity_flags_out,                sector_buffers.get_buffer(ENTITY_FLAG))
            .buf_arg(EgressEntities_k.Args.hull_bones_out,                  sector_buffers.get_buffer(HULL_BONE))
            .buf_arg(EgressEntities_k.Args.hull_bind_pose_indicies_out,     sector_buffers.get_buffer(HULL_BONE_BIND_POSE))
            .buf_arg(EgressEntities_k.Args.hull_inv_bind_pose_indicies_out, sector_buffers.get_buffer(HULL_BONE_INV_BIND_POSE))
            .buf_arg(EgressEntities_k.Args.armature_bones_out,              sector_buffers.get_buffer(ENTITY_BONE))
            .buf_arg(EgressEntities_k.Args.armature_bone_reference_ids_out, sector_buffers.get_buffer(ENTITY_BONE_REFERENCE_ID))
            .buf_arg(EgressEntities_k.Args.armature_bone_parent_ids_out,    sector_buffers.get_buffer(ENTITY_BONE_PARENT_ID))
            .ptr_arg(EgressEntities_k.Args.counters,                        ptr_egress_sizes);
    }

    public void egress(int entity_count, int[] egress_counts)
    {
        GPGPU.cl_zero_buffer(ptr_queue, ptr_egress_sizes, cl_int * 6);
        int entity_capacity        = egress_counts[0];
        int hull_capacity          = egress_counts[1];
        int point_capacity         = egress_counts[2];
        int edge_capacity          = egress_counts[3];
        int hull_bone_capacity     = egress_counts[4];
        int entity_bone_capacity   = egress_counts[5];
        sector_buffers.ensure_capacity_all(point_capacity, edge_capacity, hull_capacity, entity_capacity, hull_bone_capacity, entity_bone_capacity);
        k_egress_entities.call(arg_long(entity_count));
    }

    public void unload(UnorderedSectorBufferGroup.Raw raw_sectors, int[] counts)
    {
        sector_buffers.unload_sectors(raw_sectors, counts);
    }

    public void destroy()
    {
        p_gpu_crud.destroy();
        sector_buffers.destroy();
        GPGPU.cl_release_buffer(ptr_egress_sizes);
    }
}
