package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.ecs.systems.UnloadedSectorSlice;

import static com.controllerface.bvge.cl.CLUtils.*;

public class SectorOutputBuffer
{
    private final GPUProgram p_gpu_crud = new GPUCrud();
    private final GPUKernel k_egress_entities;

    //#region Armature Buffers

    private final ResizableBuffer b_entity_bone;
    private final ResizableBuffer b_entity_bone_reference_id;
    private final ResizableBuffer b_entity_bone_parent_id;

    //#endregion

    //#region Entity Buffers

    private final ResizableBuffer b_entity_anim_elapsed;
    private final ResizableBuffer b_entity_motion_state;
    private final ResizableBuffer b_entity_anim_index;
    private final ResizableBuffer b_entity;
    private final ResizableBuffer b_entity_flag;
    private final ResizableBuffer b_entity_root_hull;
    private final ResizableBuffer b_entity_model_id;
    private final ResizableBuffer b_entity_model_transform;
    private final ResizableBuffer b_entity_hull_table;
    private final ResizableBuffer b_entity_bone_table;
    private final ResizableBuffer b_entity_mass;

    //#endregion

    //#region Edge Buffers

    private final ResizableBuffer b_edge;
    private final ResizableBuffer b_edge_flag;
    private final ResizableBuffer b_edge_length;

    //#endregion

    //#region Hull Bone Buffers

    private final ResizableBuffer b_hull_bone_bind_pose_id;
    private final ResizableBuffer b_hull_bone_inv_bind_pose_id;
    private final ResizableBuffer b_hull_bone;

    //#endregion

    //#region Hull Buffers

    private final ResizableBuffer b_hull;
    private final ResizableBuffer b_hull_scale;
    private final ResizableBuffer b_hull_point_table;
    private final ResizableBuffer b_hull_edge_table;
    private final ResizableBuffer b_hull_flag;
    private final ResizableBuffer b_hull_entity_id;
    private final ResizableBuffer b_hull_bone_table;
    private final ResizableBuffer b_hull_friction;
    private final ResizableBuffer b_hull_restitution;
    private final ResizableBuffer b_hull_mesh_id;
    private final ResizableBuffer b_hull_uv_offset;
    private final ResizableBuffer b_hull_rotation;
    private final ResizableBuffer b_hull_integrity;

    //#endregion

    //#region Point Buffers

    private final ResizableBuffer b_point_bone_table;
    private final ResizableBuffer b_point;
    private final ResizableBuffer b_point_vertex_reference;
    private final ResizableBuffer b_point_hull_index;
    private final ResizableBuffer b_point_flag;
    private final ResizableBuffer b_point_hit_count;

    //#endregion

    private final long ptr_queue;
    private final long ptr_egress_sizes;

    public SectorOutputBuffer(long ptr_queue, GPUCoreMemory core_memory)
    {
        this.ptr_queue         = ptr_queue;
        this.ptr_egress_sizes  = GPGPU.cl_new_pinned_buffer(CLSize.cl_int * 6);

        // persistent buffers


        b_entity_bone                 = new PersistentBuffer(this.ptr_queue, CLSize.cl_float16);
        b_entity_bone_reference_id    = new PersistentBuffer(this.ptr_queue, CLSize.cl_int);
        b_entity_bone_parent_id       = new PersistentBuffer(this.ptr_queue, CLSize.cl_int);

        b_entity_anim_elapsed        = new PersistentBuffer(this.ptr_queue, CLSize.cl_float2, 1_000L);
        b_entity_motion_state        = new PersistentBuffer(this.ptr_queue, CLSize.cl_short2, 1_000L);
        b_entity_anim_index          = new PersistentBuffer(this.ptr_queue, CLSize.cl_int2, 1_000L);
        b_entity                     = new PersistentBuffer(this.ptr_queue, CLSize.cl_float4, 1_000L);
        b_entity_flag                = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
        b_entity_root_hull           = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
        b_entity_model_id            = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
        b_entity_model_transform     = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
        b_entity_hull_table          = new PersistentBuffer(this.ptr_queue, CLSize.cl_int2, 1_000L);
        b_entity_bone_table          = new PersistentBuffer(this.ptr_queue, CLSize.cl_int2, 1_000L);
        b_entity_mass                = new PersistentBuffer(this.ptr_queue, CLSize.cl_float, 1_000L);

        b_edge                       = new PersistentBuffer(this.ptr_queue, CLSize.cl_int2, 2_400L);
        b_edge_flag                  = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 2_400L);
        b_edge_length                = new PersistentBuffer(this.ptr_queue, CLSize.cl_float, 2_400L);

        b_hull_bone                  = new PersistentBuffer(this.ptr_queue, CLSize.cl_float16, 1_000L);
        b_hull_bone_bind_pose_id     = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
        b_hull_bone_inv_bind_pose_id = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);

        b_hull                       = new PersistentBuffer(this.ptr_queue, CLSize.cl_float4, 1_000L);
        b_hull_scale                 = new PersistentBuffer(this.ptr_queue, CLSize.cl_float2, 1_000L);
        b_hull_point_table           = new PersistentBuffer(this.ptr_queue, CLSize.cl_int2, 1_000L);
        b_hull_edge_table            = new PersistentBuffer(this.ptr_queue, CLSize.cl_int2, 1_000L);
        b_hull_flag                  = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
        b_hull_bone_table            = new PersistentBuffer(this.ptr_queue, CLSize.cl_int2, 1_000L);
        b_hull_entity_id             = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
        b_hull_friction              = new PersistentBuffer(this.ptr_queue, CLSize.cl_float, 1_000L);
        b_hull_restitution           = new PersistentBuffer(this.ptr_queue, CLSize.cl_float, 1_000L);
        b_hull_mesh_id               = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
        b_hull_uv_offset             = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
        b_hull_rotation              = new PersistentBuffer(this.ptr_queue, CLSize.cl_float2, 1_000L);
        b_hull_integrity             = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);

        b_point_bone_table           = new PersistentBuffer(this.ptr_queue, CLSize.cl_int4, 5_000L);
        b_point                      = new PersistentBuffer(this.ptr_queue, CLSize.cl_float4, 5_000L);
        b_point_vertex_reference     = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 5_000L);
        b_point_hull_index           = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 5_000L);
        b_point_flag                 = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 5_000L);
        b_point_hit_count            = new PersistentBuffer(this.ptr_queue, CLSize.cl_short, 5_000L);

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
            .buf_arg(EgressEntities_k.Args.armature_bones_in,               core_memory.buffer(BufferType.ARMATURE_BONE))
            .buf_arg(EgressEntities_k.Args.armature_bone_reference_ids_in,  core_memory.buffer(BufferType.ARMATURE_BONE_REFERENCE_ID))
            .buf_arg(EgressEntities_k.Args.armature_bone_parent_ids_in,     core_memory.buffer(BufferType.ARMATURE_BONE_PARENT_ID))
            .buf_arg(EgressEntities_k.Args.points_out,                      b_point)
            .buf_arg(EgressEntities_k.Args.point_vertex_references_out,     b_point_vertex_reference)
            .buf_arg(EgressEntities_k.Args.point_hull_indices_out,          b_point_hull_index)
            .buf_arg(EgressEntities_k.Args.point_hit_counts_out,            b_point_hit_count)
            .buf_arg(EgressEntities_k.Args.point_flags_out,                 b_point_flag)
            .buf_arg(EgressEntities_k.Args.point_bone_tables_out,           b_point_bone_table)
            .buf_arg(EgressEntities_k.Args.edges_out,                       b_edge)
            .buf_arg(EgressEntities_k.Args.edge_lengths_out,                b_edge_length)
            .buf_arg(EgressEntities_k.Args.edge_flags_out,                  b_edge_flag)
            .buf_arg(EgressEntities_k.Args.hulls_out,                       b_hull)
            .buf_arg(EgressEntities_k.Args.hull_scales_out,                 b_hull_scale)
            .buf_arg(EgressEntities_k.Args.hull_rotations_out,              b_hull_rotation)
            .buf_arg(EgressEntities_k.Args.hull_frictions_out,              b_hull_friction)
            .buf_arg(EgressEntities_k.Args.hull_restitutions_out,           b_hull_restitution)
            .buf_arg(EgressEntities_k.Args.hull_point_tables_out,           b_hull_point_table)
            .buf_arg(EgressEntities_k.Args.hull_edge_tables_out,            b_hull_edge_table)
            .buf_arg(EgressEntities_k.Args.hull_bone_tables_out,            b_hull_bone_table)
            .buf_arg(EgressEntities_k.Args.hull_entity_ids_out,             b_hull_entity_id)
            .buf_arg(EgressEntities_k.Args.hull_flags_out,                  b_hull_flag)
            .buf_arg(EgressEntities_k.Args.hull_mesh_ids_out,               b_hull_mesh_id)
            .buf_arg(EgressEntities_k.Args.hull_uv_offsets_out,             b_hull_uv_offset)
            .buf_arg(EgressEntities_k.Args.hull_integrity_out,              b_hull_integrity)
            .buf_arg(EgressEntities_k.Args.entities_out,                    b_entity)
            .buf_arg(EgressEntities_k.Args.entity_animation_elapsed_out,    b_entity_anim_elapsed)
            .buf_arg(EgressEntities_k.Args.entity_motion_states_out,        b_entity_motion_state)
            .buf_arg(EgressEntities_k.Args.entity_animation_indices_out,    b_entity_anim_index)
            .buf_arg(EgressEntities_k.Args.entity_hull_tables_out,          b_entity_hull_table)
            .buf_arg(EgressEntities_k.Args.entity_bone_tables_out,          b_entity_bone_table)
            .buf_arg(EgressEntities_k.Args.entity_masses_out,               b_entity_mass)
            .buf_arg(EgressEntities_k.Args.entity_root_hulls_out,           b_entity_root_hull)
            .buf_arg(EgressEntities_k.Args.entity_model_indices_out,        b_entity_model_id)
            .buf_arg(EgressEntities_k.Args.entity_model_transforms_out,     b_entity_model_transform)
            .buf_arg(EgressEntities_k.Args.entity_flags_out,                b_entity_flag)
            .buf_arg(EgressEntities_k.Args.hull_bones_out,                  b_hull_bone)
            .buf_arg(EgressEntities_k.Args.hull_bind_pose_indicies_out,     b_hull_bone_bind_pose_id)
            .buf_arg(EgressEntities_k.Args.hull_inv_bind_pose_indicies_out, b_hull_bone_inv_bind_pose_id)
            .buf_arg(EgressEntities_k.Args.armature_bones_out, b_entity_bone)
            .buf_arg(EgressEntities_k.Args.armature_bone_reference_ids_out, b_entity_bone_reference_id)
            .buf_arg(EgressEntities_k.Args.armature_bone_parent_ids_out, b_entity_bone_parent_id)
            .ptr_arg(EgressEntities_k.Args.counters,                        ptr_egress_sizes);
    }

    public void pull_from_parent(int entity_count, int[] egress_counts)
    {
        GPGPU.cl_zero_buffer(ptr_queue, ptr_egress_sizes, CLSize.cl_int * 6);

        int entity_capacity        = egress_counts[0];
        int hull_capacity          = egress_counts[1];
        int point_capacity         = egress_counts[2];
        int edge_capacity          = egress_counts[3];
        int hull_bone_capacity     = egress_counts[4];
        int entity_bone_capacity   = egress_counts[5];

        b_point.ensure_capacity(point_capacity);
        b_point_vertex_reference.ensure_capacity(point_capacity);
        b_point_hull_index.ensure_capacity(point_capacity);
        b_point_flag.ensure_capacity(point_capacity);
        b_point_hit_count.ensure_capacity(point_capacity);
        b_point_bone_table.ensure_capacity(point_capacity);

        b_edge.ensure_capacity(edge_capacity);
        b_edge_length.ensure_capacity(edge_capacity);
        b_edge_flag.ensure_capacity(edge_capacity);

        b_hull.ensure_capacity(hull_capacity);
        b_hull_scale.ensure_capacity(hull_capacity);
        b_hull_mesh_id.ensure_capacity(hull_capacity);
        b_hull_uv_offset.ensure_capacity(hull_capacity);
        b_hull_rotation.ensure_capacity(hull_capacity);
        b_hull_integrity.ensure_capacity(hull_capacity);
        b_hull_point_table.ensure_capacity(hull_capacity);
        b_hull_edge_table.ensure_capacity(hull_capacity);
        b_hull_flag.ensure_capacity(hull_capacity);
        b_hull_bone_table.ensure_capacity(hull_capacity);
        b_hull_entity_id.ensure_capacity(hull_capacity);
        b_hull_friction.ensure_capacity(hull_capacity);
        b_hull_restitution.ensure_capacity(hull_capacity);

        b_entity.ensure_capacity(entity_capacity);
        b_entity_flag.ensure_capacity(entity_capacity);
        b_entity_root_hull.ensure_capacity(entity_capacity);
        b_entity_model_id.ensure_capacity(entity_capacity);
        b_entity_model_transform.ensure_capacity(entity_capacity);
        b_entity_mass.ensure_capacity(entity_capacity);
        b_entity_anim_index.ensure_capacity(entity_capacity);
        b_entity_anim_elapsed.ensure_capacity(entity_capacity);
        b_entity_motion_state.ensure_capacity(entity_capacity);
        b_entity_hull_table.ensure_capacity(entity_capacity);
        b_entity_bone_table.ensure_capacity(entity_capacity);

        b_hull_bone.ensure_capacity(hull_bone_capacity);
        b_hull_bone_bind_pose_id.ensure_capacity(hull_bone_capacity);
        b_hull_bone_inv_bind_pose_id.ensure_capacity(hull_bone_capacity);

        b_entity_bone.ensure_capacity(entity_bone_capacity);
        b_entity_bone_reference_id.ensure_capacity(entity_bone_capacity);
        b_entity_bone_parent_id.ensure_capacity(entity_bone_capacity);

        k_egress_entities.call(arg_long(entity_count));
    }

    public void unload_sector(UnloadedSectorSlice unloaded_sectors, int[] counts)
    {
        int entity_capacity        = counts[0];
        int hull_capacity          = counts[1];
        int point_capacity         = counts[2];
        int edge_capacity          = counts[3];
        int hull_bone_capacity     = counts[4];
        int entity_bone_capacity   = counts[5];

        int entity_vec2       = entity_capacity      * 2;
        int entity_vec4       = entity_capacity      * 4;
        int hull_vec2         = hull_capacity        * 2;
        int hull_vec4         = hull_capacity        * 4;
        int point_vec4        = point_capacity       * 4;
        int edge_vec2         = edge_capacity        * 2;
        int hull_bone_vec16   = hull_bone_capacity   * 16;
        int entity_bone_vec16 = entity_bone_capacity * 16;

        if (point_capacity > 0)
        {
            b_point.transfer_out_float(unloaded_sectors.raw_point,                                         CLSize.cl_float,  point_vec4);
            b_point_vertex_reference.transfer_out_int(unloaded_sectors.raw_point_vertex_reference,         CLSize.cl_int,     point_capacity);
            b_point_hull_index.transfer_out_int(unloaded_sectors.raw_point_hull_index,                     CLSize.cl_int,     point_capacity);
            b_point_flag.transfer_out_int(unloaded_sectors.raw_point_flag,                                 CLSize.cl_int,     point_capacity);
            b_point_hit_count.transfer_out_short(unloaded_sectors.raw_point_hit_count,                     CLSize.cl_short,   point_capacity);
            b_point_bone_table.transfer_out_int(unloaded_sectors.raw_point_bone_table,                     CLSize.cl_int,    point_vec4);
        }

        if (edge_capacity > 0)
        {
            b_edge.transfer_out_int(unloaded_sectors.raw_edge,                                             CLSize.cl_int,    edge_vec2);
            b_edge_length.transfer_out_float(unloaded_sectors.raw_edge_length,                             CLSize.cl_float,   edge_capacity);
            b_edge_flag.transfer_out_int(unloaded_sectors.raw_edge_flag,                                   CLSize.cl_int,     edge_capacity);
        }

        if (hull_capacity > 0)
        {
            b_hull.transfer_out_float(unloaded_sectors.raw_hull,                                           CLSize.cl_float,  hull_vec4);
            b_hull_scale.transfer_out_float(unloaded_sectors.raw_hull_scale,                               CLSize.cl_float,  hull_vec2);
            b_hull_mesh_id.transfer_out_int(unloaded_sectors.raw_hull_mesh_id,                             CLSize.cl_int,     hull_capacity);
            b_hull_uv_offset.transfer_out_int(unloaded_sectors.raw_hull_uv_offset,                         CLSize.cl_int,     hull_capacity);
            b_hull_rotation.transfer_out_float(unloaded_sectors.raw_hull_rotation,                         CLSize.cl_float,  hull_vec2);
            b_hull_integrity.transfer_out_int(unloaded_sectors.raw_hull_integrity,                         CLSize.cl_int,     hull_capacity);
            b_hull_point_table.transfer_out_int(unloaded_sectors.raw_hull_point_table,                     CLSize.cl_int,    hull_vec2);
            b_hull_edge_table.transfer_out_int(unloaded_sectors.raw_hull_edge_table,                       CLSize.cl_int,    hull_vec2);
            b_hull_flag.transfer_out_int(unloaded_sectors.raw_hull_flag,                                   CLSize.cl_int,     hull_capacity);
            b_hull_bone_table.transfer_out_int(unloaded_sectors.raw_hull_bone_table,                       CLSize.cl_int,    hull_vec2);
            b_hull_entity_id.transfer_out_int(unloaded_sectors.raw_hull_entity_id,                         CLSize.cl_int,     hull_capacity);
            b_hull_friction.transfer_out_float(unloaded_sectors.raw_hull_friction,                         CLSize.cl_float,   hull_capacity);
            b_hull_restitution.transfer_out_float(unloaded_sectors.raw_hull_restitution,                   CLSize.cl_float,   hull_capacity);
        }

        if (entity_capacity > 0)
        {
            b_entity.transfer_out_float(unloaded_sectors.raw_entity,                                       CLSize.cl_float,  entity_vec4);
            b_entity_flag.transfer_out_int(unloaded_sectors.raw_entity_flag,                               CLSize.cl_int,     entity_capacity);
            b_entity_root_hull.transfer_out_int(unloaded_sectors.raw_entity_root_hull,                     CLSize.cl_int,     entity_capacity);
            b_entity_model_id.transfer_out_int(unloaded_sectors.raw_entity_model_id,                       CLSize.cl_int,     entity_capacity);
            b_entity_model_transform.transfer_out_int(unloaded_sectors.raw_entity_model_transform,         CLSize.cl_int,     entity_capacity);
            b_entity_mass.transfer_out_float(unloaded_sectors.raw_entity_mass,                             CLSize.cl_float,   entity_capacity);
            b_entity_anim_index.transfer_out_int(unloaded_sectors.raw_entity_anim_index,                   CLSize.cl_int,    entity_vec2);
            b_entity_anim_elapsed.transfer_out_float(unloaded_sectors.raw_entity_anim_elapsed,             CLSize.cl_float,  entity_vec2);
            b_entity_motion_state.transfer_out_short(unloaded_sectors.raw_entity_motion_state,             CLSize.cl_short,  entity_vec2);
            b_entity_hull_table.transfer_out_int(unloaded_sectors.raw_entity_hull_table,                   CLSize.cl_int,    entity_vec2);
            b_entity_bone_table.transfer_out_int(unloaded_sectors.raw_entity_bone_table,                   CLSize.cl_int,    entity_vec2);
        }

        if (hull_bone_capacity > 0)
        {
            b_hull_bone.transfer_out_float(unloaded_sectors.raw_hull_bone,                                 CLSize.cl_float, hull_bone_vec16);
            b_hull_bone_bind_pose_id.transfer_out_int(unloaded_sectors.raw_hull_bone_bind_pose_id,         CLSize.cl_int,     hull_bone_capacity);
            b_hull_bone_inv_bind_pose_id.transfer_out_int(unloaded_sectors.raw_hull_bone_inv_bind_pose_id, CLSize.cl_int,     hull_bone_capacity);
        }

        if (entity_bone_capacity > 0)
        {
            b_entity_bone.transfer_out_float(unloaded_sectors.raw_entity_bone,                             CLSize.cl_float,  entity_bone_vec16);
            b_entity_bone_reference_id.transfer_out_int(unloaded_sectors.raw_entity_bone_reference_id,     CLSize.cl_int,     entity_bone_capacity);
            b_entity_bone_parent_id.transfer_out_int(unloaded_sectors.raw_entity_bone_parent_id,           CLSize.cl_int,     entity_bone_capacity);
        }
    }

    public void destroy()
    {
        p_gpu_crud.destroy();

        b_edge.release();
        b_edge_length.release();
        b_edge_flag.release();
        b_hull.release();
        b_hull_scale.release();
        b_hull_mesh_id.release();
        b_hull_uv_offset.release();
        b_hull_rotation.release();
        b_hull_integrity.release();
        b_hull_point_table.release();
        b_hull_edge_table.release();
        b_hull_flag.release();
        b_hull_bone_table.release();
        b_hull_entity_id.release();
        b_hull_bone.release();
        b_hull_bone_bind_pose_id.release();
        b_hull_bone_inv_bind_pose_id.release();
        b_hull_friction.release();
        b_hull_restitution.release();
        b_point.release();
        b_point_vertex_reference.release();
        b_point_hull_index.release();
        b_point_flag.release();
        b_point_hit_count.release();
        b_point_bone_table.release();
        b_entity_bone.release();
        b_entity_bone_reference_id.release();
        b_entity_bone_parent_id.release();
        b_entity.release();
        b_entity_flag.release();
        b_entity_root_hull.release();
        b_entity_model_id.release();
        b_entity_model_transform.release();
        b_entity_mass.release();
        b_entity_anim_index.release();
        b_entity_anim_elapsed.release();
        b_entity_motion_state.release();
        b_entity_hull_table.release();
        b_entity_bone_table.release();

        debug();
    }

    private void debug()
    {
        long total = 0;
        total += b_edge.debug_data();
        total += b_edge_length.debug_data();
        total += b_edge_flag.debug_data();
        total += b_hull.debug_data();
        total += b_hull_scale.debug_data();
        total += b_hull_mesh_id.debug_data();
        total += b_hull_uv_offset.debug_data();
        total += b_hull_rotation.debug_data();
        total += b_hull_integrity.debug_data();
        total += b_hull_point_table.debug_data();
        total += b_hull_edge_table.debug_data();
        total += b_hull_flag.debug_data();
        total += b_hull_bone_table.debug_data();
        total += b_hull_entity_id.debug_data();
        total += b_hull_bone.debug_data();
        total += b_hull_bone_bind_pose_id.debug_data();
        total += b_hull_bone_inv_bind_pose_id.debug_data();
        total += b_hull_friction.debug_data();
        total += b_hull_restitution.debug_data();
        total += b_point.debug_data();
        total += b_point_vertex_reference.debug_data();
        total += b_point_hull_index.debug_data();
        total += b_point_flag.debug_data();
        total += b_point_hit_count.debug_data();
        total += b_point_bone_table.debug_data();
        total += b_entity_bone.debug_data();
        total += b_entity_bone_reference_id.debug_data();
        total += b_entity_bone_parent_id.debug_data();
        total += b_entity.debug_data();
        total += b_entity_flag.debug_data();
        total += b_entity_root_hull.debug_data();
        total += b_entity_model_id.debug_data();
        total += b_entity_model_transform.debug_data();
        total += b_entity_mass.debug_data();
        total += b_entity_anim_index.debug_data();
        total += b_entity_anim_elapsed.debug_data();
        total += b_entity_motion_state.debug_data();
        total += b_entity_hull_table.debug_data();
        total += b_entity_bone_table.debug_data();

        //System.out.println("---------------------------");
        System.out.println("Egress Buffer Memory Usage: MB " + ((float) total / 1024f / 1024f));
    }
}
