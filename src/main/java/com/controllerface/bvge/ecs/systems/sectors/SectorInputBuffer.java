package com.controllerface.bvge.ecs.systems.sectors;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.buffers.BufferType;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.*;
import static com.controllerface.bvge.cl.CLUtils.arg_short2;

public class SectorInputBuffer implements SectorContainer
{
    private static final long ENTITY_CAP = 1_000L;
    private static final long HULL_CAP = 1_000L;
    private static final long EDGE_CAP = 2_400L;
    private static final long POINT_CAP = 5_000L;

    private final GPUProgram p_gpu_crud;

    private final GPUKernel k_create_point;
    private final GPUKernel k_create_edge;
    private final GPUKernel k_create_hull;
    private final GPUKernel k_create_entity;
    private final GPUKernel k_create_hull_bone;
    private final GPUKernel k_create_entity_bone;

    private final GPUKernel k_merge_point;
    private final GPUKernel k_merge_edge;
    private final GPUKernel k_merge_hull;
    private final GPUKernel k_merge_entity;
    private final GPUKernel k_merge_hull_bone;
    private final GPUKernel k_merge_entity_bone;

//    //#region Armature Buffers
//
//    private final ResizableBuffer b_armature_bone;
//    private final ResizableBuffer b_armature_bone_reference_id;
//    private final ResizableBuffer b_armature_bone_parent_id;
//
//    //#endregion
//
//    //#region Entity Buffers
//
//    private final ResizableBuffer b_entity_anim_elapsed;
//    private final ResizableBuffer b_entity_motion_state;
//    private final ResizableBuffer b_entity_anim_index;
//    private final ResizableBuffer b_entity;
//    private final ResizableBuffer b_entity_flag;
//    private final ResizableBuffer b_entity_root_hull;
//    private final ResizableBuffer b_entity_model_id;
//    private final ResizableBuffer b_entity_model_transform;
//    private final ResizableBuffer b_entity_hull_table;
//    private final ResizableBuffer b_entity_bone_table;
//    private final ResizableBuffer b_entity_mass;
//
//    //#endregion
//
//    //#region Edge Buffers
//
//    private final ResizableBuffer b_edge;
//    private final ResizableBuffer b_edge_flag;
//    private final ResizableBuffer b_edge_length;
//
//    //#endregion
//
//    //#region Hull Bone Buffers
//
//    private final ResizableBuffer b_hull_bone_bind_pose_id;
//    private final ResizableBuffer b_hull_bone_inv_bind_pose_id;
//    private final ResizableBuffer b_hull_bone;
//
//    //#endregion
//
//    //#region Hull Buffers
//
//    private final ResizableBuffer b_hull;
//    private final ResizableBuffer b_hull_scale;
//    private final ResizableBuffer b_hull_point_table;
//    private final ResizableBuffer b_hull_edge_table;
//    private final ResizableBuffer b_hull_flag;
//    private final ResizableBuffer b_hull_entity_id;
//    private final ResizableBuffer b_hull_bone_table;
//    private final ResizableBuffer b_hull_friction;
//    private final ResizableBuffer b_hull_restitution;
//    private final ResizableBuffer b_hull_mesh_id;
//    private final ResizableBuffer b_hull_uv_offset;
//    private final ResizableBuffer b_hull_rotation;
//    private final ResizableBuffer b_hull_integrity;
//
//    //#endregion
//
//    //#region Point Buffers
//
//    private final ResizableBuffer b_point_bone_table;
//    private final ResizableBuffer b_point;
//    private final ResizableBuffer b_point_vertex_reference;
//    private final ResizableBuffer b_point_hull_index;
//    private final ResizableBuffer b_point_flag;
//    private final ResizableBuffer b_point_hit_count;

    //#endregion

    private int point_index           = 0;
    private int edge_index            = 0;
    private int hull_index            = 0;
    private int entity_index          = 0;
    private int hull_bone_index       = 0;
    private int armature_bone_index   = 0;

    private final long ptr_queue;
    private final OrderedSectorGroup sector_group;

    public SectorInputBuffer(long ptr_queue, GPUProgram p_gpu_crud, GPUCoreMemory core_memory)
    {
        this.ptr_queue  = ptr_queue;
        this.p_gpu_crud = p_gpu_crud;
        this.sector_group = new OrderedSectorGroup(this.ptr_queue, ENTITY_CAP, HULL_CAP, EDGE_CAP, POINT_CAP);

        // persistent buffers
//        b_entity_anim_elapsed        = new PersistentBuffer(this.ptr_queue, CLSize.cl_float2, 1_000L);
//        b_entity_motion_state        = new PersistentBuffer(this.ptr_queue, CLSize.cl_short2, 1_000L);
//        b_entity_anim_index          = new PersistentBuffer(this.ptr_queue, CLSize.cl_int2, 1_000L);
//        b_armature_bone              = new PersistentBuffer(this.ptr_queue, CLSize.cl_float16);
//        b_armature_bone_reference_id = new PersistentBuffer(this.ptr_queue, CLSize.cl_int);
//        b_armature_bone_parent_id    = new PersistentBuffer(this.ptr_queue, CLSize.cl_int);
//        b_entity                     = new PersistentBuffer(this.ptr_queue, CLSize.cl_float4, 1_000L);
//        b_entity_flag                = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
//        b_entity_root_hull           = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
//        b_entity_model_id            = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
//        b_entity_model_transform     = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
//        b_entity_hull_table          = new PersistentBuffer(this.ptr_queue, CLSize.cl_int2, 1_000L);
//        b_entity_bone_table          = new PersistentBuffer(this.ptr_queue, CLSize.cl_int2, 1_000L);
//        b_entity_mass                = new PersistentBuffer(this.ptr_queue, CLSize.cl_float, 1_000L);
//        b_edge                       = new PersistentBuffer(this.ptr_queue, CLSize.cl_int2, 2_400L);
//        b_edge_flag                  = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 2_400L);
//        b_edge_length                = new PersistentBuffer(this.ptr_queue, CLSize.cl_float, 2_400L);
//        b_hull_bone                  = new PersistentBuffer(this.ptr_queue, CLSize.cl_float16, 1_000L);
//        b_hull_bone_bind_pose_id     = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
//        b_hull_bone_inv_bind_pose_id = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
//        b_hull                       = new PersistentBuffer(this.ptr_queue, CLSize.cl_float4, 1_000L);
//        b_hull_scale                 = new PersistentBuffer(this.ptr_queue, CLSize.cl_float2, 1_000L);
//        b_hull_point_table           = new PersistentBuffer(this.ptr_queue, CLSize.cl_int2, 1_000L);
//        b_hull_edge_table            = new PersistentBuffer(this.ptr_queue, CLSize.cl_int2, 1_000L);
//        b_hull_flag                  = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
//        b_hull_bone_table            = new PersistentBuffer(this.ptr_queue, CLSize.cl_int2, 1_000L);
//        b_hull_entity_id             = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
//        b_hull_friction              = new PersistentBuffer(this.ptr_queue, CLSize.cl_float, 1_000L);
//        b_hull_restitution           = new PersistentBuffer(this.ptr_queue, CLSize.cl_float, 1_000L);
//        b_hull_mesh_id               = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
//        b_hull_uv_offset             = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
//        b_hull_rotation              = new PersistentBuffer(this.ptr_queue, CLSize.cl_float2, 1_000L);
//        b_hull_integrity             = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 1_000L);
//        b_point_bone_table           = new PersistentBuffer(this.ptr_queue, CLSize.cl_int4, 5_000L);
//        b_point                      = new PersistentBuffer(this.ptr_queue, CLSize.cl_float4, 5_000L);
//        b_point_vertex_reference     = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 5_000L);
//        b_point_hull_index           = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 5_000L);
//        b_point_flag                 = new PersistentBuffer(this.ptr_queue, CLSize.cl_int, 5_000L);
//        b_point_hit_count            = new PersistentBuffer(this.ptr_queue, CLSize.cl_short, 5_000L);

        long k_ptr_create_point = p_gpu_crud.kernel_ptr(Kernel.create_point);
        k_create_point = new CreatePoint_k(this.ptr_queue, k_ptr_create_point)
            .buf_arg(CreatePoint_k.Args.points, sector_group.buffer(BufferType.POINT))
            .buf_arg(CreatePoint_k.Args.point_vertex_references, sector_group.buffer(BufferType.POINT_VERTEX_REFERENCE))
            .buf_arg(CreatePoint_k.Args.point_hull_indices, sector_group.buffer(BufferType.POINT_HULL_INDEX))
            .buf_arg(CreatePoint_k.Args.point_hit_counts, sector_group.buffer(BufferType.POINT_HIT_COUNT))
            .buf_arg(CreatePoint_k.Args.point_flags, sector_group.buffer(BufferType.POINT_FLAG))
            .buf_arg(CreatePoint_k.Args.point_bone_tables, sector_group.buffer(BufferType.POINT_BONE_TABLE));

        long k_ptr_create_edge = p_gpu_crud.kernel_ptr(Kernel.create_edge);
        k_create_edge = new CreateEdge_k(this.ptr_queue, k_ptr_create_edge)
            .buf_arg(CreateEdge_k.Args.edges, sector_group.buffer(BufferType.EDGE))
            .buf_arg(CreateEdge_k.Args.edge_lengths, sector_group.buffer(BufferType.EDGE_LENGTH))
            .buf_arg(CreateEdge_k.Args.edge_flags, sector_group.buffer(BufferType.EDGE_FLAG));

        long k_ptr_create_hull = p_gpu_crud.kernel_ptr(Kernel.create_hull);
        k_create_hull = new CreateHull_k(this.ptr_queue, k_ptr_create_hull)
            .buf_arg(CreateHull_k.Args.hulls, sector_group.buffer(BufferType.HULL))
            .buf_arg(CreateHull_k.Args.hull_scales, sector_group.buffer(BufferType.HULL_SCALE))
            .buf_arg(CreateHull_k.Args.hull_rotations, sector_group.buffer(BufferType.HULL_ROTATION))
            .buf_arg(CreateHull_k.Args.hull_frictions, sector_group.buffer(BufferType.HULL_FRICTION))
            .buf_arg(CreateHull_k.Args.hull_restitutions, sector_group.buffer(BufferType.HULL_RESTITUTION))
            .buf_arg(CreateHull_k.Args.hull_point_tables, sector_group.buffer(BufferType.HULL_POINT_TABLE))
            .buf_arg(CreateHull_k.Args.hull_edge_tables, sector_group.buffer(BufferType.HULL_EDGE_TABLE))
            .buf_arg(CreateHull_k.Args.hull_bone_tables, sector_group.buffer(BufferType.HULL_BONE_TABLE))
            .buf_arg(CreateHull_k.Args.hull_entity_ids, sector_group.buffer(BufferType.HULL_ENTITY_ID))
            .buf_arg(CreateHull_k.Args.hull_flags, sector_group.buffer(BufferType.HULL_FLAG))
            .buf_arg(CreateHull_k.Args.hull_mesh_ids, sector_group.buffer(BufferType.HULL_MESH_ID))
            .buf_arg(CreateHull_k.Args.hull_uv_offsets, sector_group.buffer(BufferType.HULL_UV_OFFSET))
            .buf_arg(CreateHull_k.Args.hull_integrity, sector_group.buffer(BufferType.HULL_INTEGRITY));

        long k_ptr_create_entity = p_gpu_crud.kernel_ptr(Kernel.create_entity);
        k_create_entity = new CreateEntity_k(this.ptr_queue, k_ptr_create_entity)
            .buf_arg(CreateEntity_k.Args.entities, sector_group.buffer(BufferType.ENTITY))
            .buf_arg(CreateEntity_k.Args.entity_root_hulls, sector_group.buffer(BufferType.ENTITY_ROOT_HULL))
            .buf_arg(CreateEntity_k.Args.entity_model_indices, sector_group.buffer(BufferType.ENTITY_MODEL_ID))
            .buf_arg(CreateEntity_k.Args.entity_model_transforms, sector_group.buffer(BufferType.ENTITY_TRANSFORM_ID))
            .buf_arg(CreateEntity_k.Args.entity_flags, sector_group.buffer(BufferType.ENTITY_FLAG))
            .buf_arg(CreateEntity_k.Args.entity_hull_tables, sector_group.buffer(BufferType.ENTITY_HULL_TABLE))
            .buf_arg(CreateEntity_k.Args.entity_bone_tables, sector_group.buffer(BufferType.ENTITY_BONE_TABLE))
            .buf_arg(CreateEntity_k.Args.entity_masses, sector_group.buffer(BufferType.ENTITY_MASS))
            .buf_arg(CreateEntity_k.Args.entity_animation_indices, sector_group.buffer(BufferType.ENTITY_ANIM_INDEX))
            .buf_arg(CreateEntity_k.Args.entity_animation_elapsed, sector_group.buffer(BufferType.ENTITY_ANIM_ELAPSED))
            .buf_arg(CreateEntity_k.Args.entity_motion_states, sector_group.buffer(BufferType.ENTITY_MOTION_STATE));

        long k_ptr_create_hull_bone = p_gpu_crud.kernel_ptr(Kernel.create_hull_bone);
        k_create_hull_bone = new CreateHullBone_k(this.ptr_queue, k_ptr_create_hull_bone)
            .buf_arg(CreateHullBone_k.Args.hull_bones, sector_group.buffer(BufferType.HULL_BONE))
            .buf_arg(CreateHullBone_k.Args.hull_bind_pose_indicies, sector_group.buffer(BufferType.HULL_BONE_BIND_POSE))
            .buf_arg(CreateHullBone_k.Args.hull_inv_bind_pose_indicies, sector_group.buffer(BufferType.HULL_BONE_INV_BIND_POSE));

        long k_ptr_create_entity_bone = p_gpu_crud.kernel_ptr(Kernel.create_entity_bone);
        k_create_entity_bone = new CreateEntityBone_k(this.ptr_queue, k_ptr_create_entity_bone)
            .buf_arg(CreateEntityBone_k.Args.entity_bones, sector_group.buffer(BufferType.ENTITY_BONE))
            .buf_arg(CreateEntityBone_k.Args.entity_bone_reference_ids, sector_group.buffer(BufferType.ENTITY_BONE_REFERENCE_ID))
            .buf_arg(CreateEntityBone_k.Args.entity_bone_parent_ids, sector_group.buffer(BufferType.ENTITY_BONE_PARENT_ID));

        long k_ptr_merge_point = p_gpu_crud.kernel_ptr(Kernel.merge_point);
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

        long k_ptr_merge_edge = p_gpu_crud.kernel_ptr(Kernel.merge_edge);
        k_merge_edge = new MergeEdge_k(this.ptr_queue, k_ptr_merge_edge)
            .buf_arg(MergeEdge_k.Args.edges_in, sector_group.buffer(BufferType.EDGE))
            .buf_arg(MergeEdge_k.Args.edge_lengths_in, sector_group.buffer(BufferType.EDGE_LENGTH))
            .buf_arg(MergeEdge_k.Args.edge_flags_in, sector_group.buffer(BufferType.EDGE_FLAG))
            .buf_arg(MergeEdge_k.Args.edges_out, core_memory.buffer(BufferType.EDGE))
            .buf_arg(MergeEdge_k.Args.edge_lengths_out, core_memory.buffer(BufferType.EDGE_LENGTH))
            .buf_arg(MergeEdge_k.Args.edge_flags_out, core_memory.buffer(BufferType.EDGE_FLAG));

        long k_ptr_merge_hull = p_gpu_crud.kernel_ptr(Kernel.merge_hull);
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

        long k_ptr_merge_entity = p_gpu_crud.kernel_ptr(Kernel.merge_entity);
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

        long k_ptr_merge_hull_bone = p_gpu_crud.kernel_ptr(Kernel.merge_hull_bone);
        k_merge_hull_bone = new MergeHullBone_k(this.ptr_queue, k_ptr_merge_hull_bone)
            .buf_arg(MergeHullBone_k.Args.hull_bones_in, sector_group.buffer(BufferType.HULL_BONE))
            .buf_arg(MergeHullBone_k.Args.hull_bind_pose_indicies_in, sector_group.buffer(BufferType.HULL_BONE_BIND_POSE))
            .buf_arg(MergeHullBone_k.Args.hull_inv_bind_pose_indicies_in, sector_group.buffer(BufferType.HULL_BONE_INV_BIND_POSE))
            .buf_arg(MergeHullBone_k.Args.hull_bones_out, core_memory.buffer(BufferType.HULL_BONE))
            .buf_arg(MergeHullBone_k.Args.hull_bind_pose_indicies_out, core_memory.buffer(BufferType.HULL_BONE_BIND_POSE))
            .buf_arg(MergeHullBone_k.Args.hull_inv_bind_pose_indicies_out, core_memory.buffer(BufferType.HULL_BONE_INV_BIND_POSE));

        long k_ptr_merge_entity_bone = p_gpu_crud.kernel_ptr(Kernel.merge_entity_bone);
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

    @Override
    public int next_point()
    {
        return point_index;
    }

    @Override
    public int next_edge()
    {
        return edge_index;
    }

    @Override
    public int next_hull()
    {
        return hull_index;
    }

    @Override
    public int next_entity()
    {
        return entity_index;
    }

    @Override
    public int next_hull_bone()
    {
        return hull_bone_index;
    }

    @Override
    public int next_armature_bone()
    {
        return armature_bone_index;
    }

    @Override
    public int new_point(float[] position, int[] bone_ids, int vertex_index, int hull_index, int hit_count, int flags)
    {
        int capacity = point_index + 1;
        sector_group.buffer(BufferType.POINT).ensure_capacity(capacity);
        sector_group.buffer(BufferType.POINT_VERTEX_REFERENCE).ensure_capacity(capacity);
        sector_group.buffer(BufferType.POINT_HULL_INDEX).ensure_capacity(capacity);
        sector_group.buffer(BufferType.POINT_FLAG).ensure_capacity(capacity);
        sector_group.buffer(BufferType.POINT_HIT_COUNT).ensure_capacity(capacity);
        sector_group.buffer(BufferType.POINT_BONE_TABLE).ensure_capacity(capacity);

        var new_point = position.length == 2
            ? arg_float4(position[0], position[1], position[0], position[1])
            : position;

        k_create_point
            .set_arg(CreatePoint_k.Args.target, point_index)
            .set_arg(CreatePoint_k.Args.new_point, new_point)
            .set_arg(CreatePoint_k.Args.new_point_vertex_reference, vertex_index)
            .set_arg(CreatePoint_k.Args.new_point_hull_index, hull_index)
            .set_arg(CreatePoint_k.Args.new_point_hit_count, (short) hit_count)
            .set_arg(CreatePoint_k.Args.new_point_flags, flags)
            .set_arg(CreatePoint_k.Args.new_point_bone_table, bone_ids)
            .call(GPGPU.global_single_size);

        return point_index++;
    }

    @Override
    public int new_edge(int p1, int p2, float l, int flags)
    {
        int required_capacity = edge_index + 1;
        sector_group.buffer(BufferType.EDGE).ensure_capacity(required_capacity);
        sector_group.buffer(BufferType.EDGE_LENGTH).ensure_capacity(required_capacity);
        sector_group.buffer(BufferType.EDGE_FLAG).ensure_capacity(required_capacity);

        k_create_edge
            .set_arg(CreateEdge_k.Args.target, edge_index)
            .set_arg(CreateEdge_k.Args.new_edge, arg_int2(p1, p2))
            .set_arg(CreateEdge_k.Args.new_edge_length, l)
            .set_arg(CreateEdge_k.Args.new_edge_flag, flags)
            .call(GPGPU.global_single_size);

        return edge_index++;
    }

    @Override
    public int new_hull(int mesh_id, float[] position, float[] scale, float[] rotation, int[] point_table, int[] edge_table, int[] bone_table, float friction, float restitution, int entity_id, int uv_offset, int flags)
    {
        int capacity = hull_index + 1;
        sector_group.buffer(BufferType.HULL).ensure_capacity(capacity);
        sector_group.buffer(BufferType.HULL_SCALE).ensure_capacity(capacity);
        sector_group.buffer(BufferType.HULL_MESH_ID).ensure_capacity(capacity);
        sector_group.buffer(BufferType.HULL_UV_OFFSET).ensure_capacity(capacity);
        sector_group.buffer(BufferType.HULL_ROTATION).ensure_capacity(capacity);
        sector_group.buffer(BufferType.HULL_INTEGRITY).ensure_capacity(capacity);
        sector_group.buffer(BufferType.HULL_POINT_TABLE).ensure_capacity(capacity);
        sector_group.buffer(BufferType.HULL_EDGE_TABLE).ensure_capacity(capacity);
        sector_group.buffer(BufferType.HULL_FLAG).ensure_capacity(capacity);
        sector_group.buffer(BufferType.HULL_BONE_TABLE).ensure_capacity(capacity);
        sector_group.buffer(BufferType.HULL_ENTITY_ID).ensure_capacity(capacity);
        sector_group.buffer(BufferType.HULL_FRICTION).ensure_capacity(capacity);
        sector_group.buffer(BufferType.HULL_RESTITUTION).ensure_capacity(capacity);

        var new_hull = position.length == 2
            ? arg_float4(position[0], position[1], position[0], position[1])
            : position;

        k_create_hull
            .set_arg(CreateHull_k.Args.target, hull_index)
            .set_arg(CreateHull_k.Args.new_hull, new_hull)
            .set_arg(CreateHull_k.Args.new_hull_scale, scale)
            .set_arg(CreateHull_k.Args.new_rotation, rotation)
            .set_arg(CreateHull_k.Args.new_friction, friction)
            .set_arg(CreateHull_k.Args.new_restitution, restitution)
            .set_arg(CreateHull_k.Args.new_point_table, point_table)
            .set_arg(CreateHull_k.Args.new_edge_table, edge_table)
            .set_arg(CreateHull_k.Args.new_bone_table, bone_table)
            .set_arg(CreateHull_k.Args.new_entity_id, entity_id)
            .set_arg(CreateHull_k.Args.new_flags, flags)
            .set_arg(CreateHull_k.Args.new_hull_mesh_id, mesh_id)
            .set_arg(CreateHull_k.Args.new_hull_uv_offset, uv_offset)
            .set_arg(CreateHull_k.Args.new_hull_integrity, 100)
            .call(GPGPU.global_single_size);

        return hull_index++;
    }

    @Override
    public int new_entity(float x, float y, float z, float w, int[] hull_table, int[] bone_table, float mass, int anim_index, float anim_time, int root_hull, int model_id, int model_transform_id, int flags)
    {
        int capacity = entity_index + 1;
        sector_group.buffer(BufferType.ENTITY).ensure_capacity(capacity);
        sector_group.buffer(BufferType.ENTITY_FLAG).ensure_capacity(capacity);
        sector_group.buffer(BufferType.ENTITY_ROOT_HULL).ensure_capacity(capacity);
        sector_group.buffer(BufferType.ENTITY_MODEL_ID).ensure_capacity(capacity);
        sector_group.buffer(BufferType.ENTITY_TRANSFORM_ID).ensure_capacity(capacity);
        sector_group.buffer(BufferType.ENTITY_MASS).ensure_capacity(capacity);
        sector_group.buffer(BufferType.ENTITY_ANIM_INDEX).ensure_capacity(capacity);
        sector_group.buffer(BufferType.ENTITY_ANIM_ELAPSED).ensure_capacity(capacity);
        sector_group.buffer(BufferType.ENTITY_MOTION_STATE).ensure_capacity(capacity);
        sector_group.buffer(BufferType.ENTITY_HULL_TABLE).ensure_capacity(capacity);
        sector_group.buffer(BufferType.ENTITY_BONE_TABLE).ensure_capacity(capacity);

        k_create_entity
            .set_arg(CreateEntity_k.Args.target, entity_index)
            .set_arg(CreateEntity_k.Args.new_entity, arg_float4(x, y, z, w))
            .set_arg(CreateEntity_k.Args.new_entity_root_hull, root_hull)
            .set_arg(CreateEntity_k.Args.new_entity_model_id, model_id)
            .set_arg(CreateEntity_k.Args.new_entity_model_transform, model_transform_id)
            .set_arg(CreateEntity_k.Args.new_entity_flags, flags)
            .set_arg(CreateEntity_k.Args.new_entity_hull_table, hull_table)
            .set_arg(CreateEntity_k.Args.new_entity_bone_table, bone_table)
            .set_arg(CreateEntity_k.Args.new_entity_mass, mass)
            .set_arg(CreateEntity_k.Args.new_entity_animation_index, arg_int2(anim_index, -1))
            .set_arg(CreateEntity_k.Args.new_entity_animation_time, arg_float2(0.0f, 0.0f)) // todo: maybe remove these zero init ones
            .set_arg(CreateEntity_k.Args.new_entity_animation_state, arg_short2((short) 0, (short) 0))
            .call(GPGPU.global_single_size);

        return entity_index++;
    }

    @Override
    public int new_hull_bone(float[] bone_data, int bind_pose_id, int inv_bind_pose_id)
    {
        int capacity = hull_bone_index + 1;
        sector_group.buffer(BufferType.HULL_BONE).ensure_capacity(capacity);
        sector_group.buffer(BufferType.HULL_BONE_BIND_POSE).ensure_capacity(capacity);
        sector_group.buffer(BufferType.HULL_BONE_INV_BIND_POSE).ensure_capacity(capacity);

        k_create_hull_bone
            .set_arg(CreateHullBone_k.Args.target, hull_bone_index)
            .set_arg(CreateHullBone_k.Args.new_hull_bone, bone_data)
            .set_arg(CreateHullBone_k.Args.new_hull_bind_pose_id, bind_pose_id)
            .set_arg(CreateHullBone_k.Args.new_hull_inv_bind_pose_id, inv_bind_pose_id)
            .call(GPGPU.global_single_size);

        return hull_bone_index++;
    }

    @Override
    public int new_armature_bone(int bone_reference, int bone_parent_id, float[] bone_data)
    {
        int capacity = armature_bone_index + 1;
        sector_group.buffer(BufferType.ENTITY_BONE).ensure_capacity(capacity);
        sector_group.buffer(BufferType.ENTITY_BONE_REFERENCE_ID).ensure_capacity(capacity);
        sector_group.buffer(BufferType.ENTITY_BONE_PARENT_ID).ensure_capacity(capacity);

        k_create_entity_bone
            .set_arg(CreateEntityBone_k.Args.target, armature_bone_index)
            .set_arg(CreateEntityBone_k.Args.new_armature_bone, bone_data)
            .set_arg(CreateEntityBone_k.Args.new_armature_bone_reference, bone_reference)
            .set_arg(CreateEntityBone_k.Args.new_armature_bone_parent_id, bone_parent_id)
            .call(GPGPU.global_single_size);

        return armature_bone_index++;
    }

    @Override
    public void merge_into_parent(SectorContainer parent)
    {
        if (point_index > 0) k_merge_point
            .set_arg(MergePoint_k.Args.point_offset, parent.next_point())
            .set_arg(MergePoint_k.Args.bone_offset, parent.next_hull_bone())
            .set_arg(MergePoint_k.Args.hull_offset, parent.next_hull())
            .call(arg_long(point_index));

        if (edge_index > 0) k_merge_edge
            .set_arg(MergeEdge_k.Args.edge_offset, parent.next_edge())
            .set_arg(MergeEdge_k.Args.point_offset, parent.next_point())
            .call(arg_long(edge_index));

        if (hull_index > 0) k_merge_hull
            .set_arg(MergeHull_k.Args.hull_offset, parent.next_hull())
            .set_arg(MergeHull_k.Args.point_offset, parent.next_point())
            .set_arg(MergeHull_k.Args.edge_offset, parent.next_edge())
            .set_arg(MergeHull_k.Args.entity_offset, parent.next_entity())
            .set_arg(MergeHull_k.Args.hull_bone_offset, parent.next_hull_bone())
            .call(arg_long(hull_index));

        if (entity_index > 0) k_merge_entity
            .set_arg(MergeEntity_k.Args.entity_offset, parent.next_entity())
            .set_arg(MergeEntity_k.Args.hull_offset, parent.next_hull())
            .set_arg(MergeEntity_k.Args.armature_bone_offset, parent.next_armature_bone())
            .call(arg_long(entity_index));

        if (hull_bone_index > 0) k_merge_hull_bone
            .set_arg(MergeHullBone_k.Args.hull_bone_offset, parent.next_hull_bone())
            .set_arg(MergeHullBone_k.Args.armature_bone_offset, parent.next_armature_bone())
            .call(arg_long(hull_bone_index));

        if (armature_bone_index > 0) k_merge_entity_bone
            .set_arg(MergeEntityBone_k.Args.armature_bone_offset, parent.next_armature_bone())
            .call(arg_long(armature_bone_index));

        point_index           = 0;
        edge_index            = 0;
        hull_index            = 0;
        entity_index          = 0;
        hull_bone_index       = 0;
        armature_bone_index   = 0;
    }

    @Override
    public void destroy()
    {
        p_gpu_crud.destroy();

        sector_group.destroy();
    }
}
