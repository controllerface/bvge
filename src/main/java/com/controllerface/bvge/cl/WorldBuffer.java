package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GPUCrud;
import org.checkerframework.checker.units.qual.K;

import static com.controllerface.bvge.cl.CLUtils.*;
import static com.controllerface.bvge.cl.CLUtils.arg_short2;

public class WorldBuffer implements WorldContainer
{
    private final GPUProgram gpu_crud = new GPUCrud();

    private final GPUKernel create_point_k;
    private final GPUKernel create_edge_k;
    private final GPUKernel create_hull_k;
    private final GPUKernel create_entity_k;
    private final GPUKernel create_bone_k;
    private final GPUKernel create_armature_bone_k;
    private final GPUKernel merge_point_k;
    private final GPUKernel merge_edge_k;
    private final GPUKernel merge_hull_k;
    private final GPUKernel merge_entity_k;
    private final GPUKernel merge_hull_bone_k;
    private final GPUKernel merge_armature_bone_k;

    //#region Point Buffers

    private final ResizableBuffer point_anti_gravity_buffer;
    private final ResizableBuffer point_bone_table_buffer;
    private final ResizableBuffer point_buffer;
    private final ResizableBuffer point_vertex_reference_buffer;
    private final ResizableBuffer point_hull_index_buffer;
    private final ResizableBuffer point_flag_buffer;
    private final ResizableBuffer point_hit_count_buffer;

    //#endregion

    //#region Edge Buffers

    private final ResizableBuffer edge_buffer;
    private final ResizableBuffer edge_flag_buffer;
    private final ResizableBuffer edge_length_buffer;

    //#endregion

    //#region Hull Buffers

    private final ResizableBuffer hull_aabb_b;
    private final ResizableBuffer hull_aabb_index_b;
    private final ResizableBuffer hull_aabb_key_b;
    private final ResizableBuffer hull_bone_b;
    private final ResizableBuffer hull_bone_bind_pose_id_b;
    private final ResizableBuffer hull_bone_inv_bind_pose_id_b;
    private final ResizableBuffer hull_b;
    private final ResizableBuffer hull_scale_b;
    private final ResizableBuffer hull_point_table_b;
    private final ResizableBuffer hull_edge_table_b;
    private final ResizableBuffer hull_flag_b;
    private final ResizableBuffer hull_entity_id_b;
    private final ResizableBuffer hull_bone_table_b;
    private final ResizableBuffer hull_friction_b;
    private final ResizableBuffer hull_restitution_b;
    private final ResizableBuffer hull_mesh_id_b;
    private final ResizableBuffer hull_uv_offset_b;
    private final ResizableBuffer hull_rotation_b;
    private final ResizableBuffer hull_integrity_b;

    //#endregion

    //#region Entity Buffers

    private final ResizableBuffer entity_accel_buffer;
    private final ResizableBuffer entity_anim_elapsed_buffer;
    private final ResizableBuffer entity_anim_blend_buffer;
    private final ResizableBuffer entity_motion_state_buffer;
    private final ResizableBuffer entity_anim_index_buffer;
    private final ResizableBuffer entity_buffer;
    private final ResizableBuffer entity_flag_buffer;
    private final ResizableBuffer entity_root_hull_buffer;
    private final ResizableBuffer entity_model_id_buffer;
    private final ResizableBuffer entity_model_transform_buffer;
    private final ResizableBuffer entity_hull_table_buffer;
    private final ResizableBuffer entity_bone_table_buffer;
    private final ResizableBuffer entity_mass_buffer;

    //#endregion

    //#region Bone Buffers

    private final ResizableBuffer bone_anim_channel_table_buffer;
    private final ResizableBuffer bone_bind_pose_buffer;
    private final ResizableBuffer bone_reference_buffer;

    //#endregion

    //#region Armature Buffers

    private final ResizableBuffer armature_bone_buffer;
    private final ResizableBuffer armature_bone_reference_id_buffer;
    private final ResizableBuffer armature_bone_parent_id_buffer;

    //#endregion

    private int point_index = 0;
    private int edge_index = 0;
    private int hull_index = 0;
    private int entity_index = 0;
    private int hull_bone_index = 0;
    private int armature_bone_index = 0;

    private final WorldContainer parent;

    public WorldBuffer(GPUCoreMemory parent)
    {
        this.parent = parent;
        gpu_crud.init();

        entity_accel_buffer               = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        entity_anim_elapsed_buffer        = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        entity_anim_blend_buffer          = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        entity_motion_state_buffer        = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_short2, 10_000L);
        entity_anim_index_buffer          = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        armature_bone_buffer              = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float16);
        armature_bone_reference_id_buffer = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int);
        armature_bone_parent_id_buffer    = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int);
        entity_buffer                     = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float4, 10_000L);
        entity_flag_buffer                = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        entity_root_hull_buffer           = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        entity_model_id_buffer            = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        entity_model_transform_buffer     = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        entity_hull_table_buffer          = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        entity_bone_table_buffer          = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        entity_mass_buffer                = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float, 10_000L);
        bone_anim_channel_table_buffer    = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int2);
        bone_bind_pose_buffer             = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float16);
        bone_reference_buffer             = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float16);
        edge_buffer                       = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int2, 24_000L);
        edge_flag_buffer                  = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int, 24_000L);
        edge_length_buffer                = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float, 24_000L);
        hull_aabb_b                       = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float4, 10_000L);
        hull_aabb_index_b                 = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int4, 10_000L);
        hull_aabb_key_b                   = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        hull_bone_b                       = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float16, 10_000L);
        hull_bone_bind_pose_id_b          = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_bone_inv_bind_pose_id_b      = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_b                            = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float4, 10_000L);
        hull_scale_b                      = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        hull_point_table_b                = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        hull_edge_table_b                 = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        hull_flag_b                       = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_bone_table_b                 = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        hull_entity_id_b                  = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_friction_b                   = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float, 10_000L);
        hull_restitution_b                = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float, 10_000L);
        hull_mesh_id_b                    = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_uv_offset_b                  = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_rotation_b                   = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        hull_integrity_b                  = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        point_anti_gravity_buffer         = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float, 50_000L);
        point_bone_table_buffer           = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int4, 50_000L);
        point_buffer                      = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_float4, 50_000L);
        point_vertex_reference_buffer     = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int, 50_000L);
        point_hull_index_buffer           = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int, 50_000L);
        point_flag_buffer                 = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_int, 50_000L);
        point_hit_count_buffer            = new PersistentBuffer(GPGPU.sector_cmd_queue_ptr, CLSize.cl_ushort, 50_000L);


        long create_point_k_ptr = gpu_crud.kernel_ptr(Kernel.create_point);
        create_point_k = new CreatePoint_k(GPGPU.sector_cmd_queue_ptr, create_point_k_ptr)
            .buf_arg(CreatePoint_k.Args.points, point_buffer)
            .buf_arg(CreatePoint_k.Args.point_vertex_references, point_vertex_reference_buffer)
            .buf_arg(CreatePoint_k.Args.point_hull_indices, point_hull_index_buffer)
            .buf_arg(CreatePoint_k.Args.point_hit_counts, point_hit_count_buffer)
            .buf_arg(CreatePoint_k.Args.point_flags, point_flag_buffer)
            .buf_arg(CreatePoint_k.Args.bone_tables, point_bone_table_buffer);

        long create_edge_k_ptr = gpu_crud.kernel_ptr(Kernel.create_edge);
        create_edge_k = new CreateEdge_k(GPGPU.sector_cmd_queue_ptr, create_edge_k_ptr)
            .buf_arg(CreateEdge_k.Args.edges, edge_buffer)
            .buf_arg(CreateEdge_k.Args.edge_lengths, edge_length_buffer)
            .buf_arg(CreateEdge_k.Args.edge_flags, edge_flag_buffer);

        long create_hull_k_ptr = gpu_crud.kernel_ptr(Kernel.create_hull);
        create_hull_k = new CreateHull_k(GPGPU.sector_cmd_queue_ptr, create_hull_k_ptr)
            .buf_arg(CreateHull_k.Args.hulls, hull_b)
            .buf_arg(CreateHull_k.Args.hull_scales, hull_scale_b)
            .buf_arg(CreateHull_k.Args.hull_rotations, hull_rotation_b)
            .buf_arg(CreateHull_k.Args.hull_frictions, hull_friction_b)
            .buf_arg(CreateHull_k.Args.hull_restitutions, hull_restitution_b)
            .buf_arg(CreateHull_k.Args.hull_point_tables, hull_point_table_b)
            .buf_arg(CreateHull_k.Args.hull_edge_tables, hull_edge_table_b)
            .buf_arg(CreateHull_k.Args.hull_bone_tables, hull_bone_table_b)
            .buf_arg(CreateHull_k.Args.hull_entity_ids, hull_entity_id_b)
            .buf_arg(CreateHull_k.Args.hull_flags, hull_flag_b)
            .buf_arg(CreateHull_k.Args.hull_mesh_ids, hull_mesh_id_b)
            .buf_arg(CreateHull_k.Args.hull_uv_offsets, hull_uv_offset_b)
            .buf_arg(CreateHull_k.Args.hull_integrity, hull_integrity_b);

        long create_entity_k_ptr = gpu_crud.kernel_ptr(Kernel.create_entity);
        create_entity_k = new CreateEntity_k(GPGPU.sector_cmd_queue_ptr, create_entity_k_ptr)
            .buf_arg(CreateEntity_k.Args.entities, entity_buffer)
            .buf_arg(CreateEntity_k.Args.entity_root_hulls, entity_root_hull_buffer)
            .buf_arg(CreateEntity_k.Args.entity_model_indices, entity_model_id_buffer)
            .buf_arg(CreateEntity_k.Args.entity_model_transforms, entity_model_transform_buffer)
            .buf_arg(CreateEntity_k.Args.entity_flags, entity_flag_buffer)
            .buf_arg(CreateEntity_k.Args.entity_hull_tables, entity_hull_table_buffer)
            .buf_arg(CreateEntity_k.Args.entity_bone_tables, entity_bone_table_buffer)
            .buf_arg(CreateEntity_k.Args.entity_masses, entity_mass_buffer)
            .buf_arg(CreateEntity_k.Args.entity_animation_indices, entity_anim_index_buffer)
            .buf_arg(CreateEntity_k.Args.entity_animation_elapsed, entity_anim_elapsed_buffer)
            .buf_arg(CreateEntity_k.Args.entity_motion_states, entity_motion_state_buffer);

        long create_bone_k_ptr = gpu_crud.kernel_ptr(Kernel.create_hull_bone);
        create_bone_k = new CreateHullBone_k(GPGPU.sector_cmd_queue_ptr, create_bone_k_ptr)
            .buf_arg(CreateHullBone_k.Args.bones, hull_bone_b)
            .buf_arg(CreateHullBone_k.Args.hull_bind_pose_indicies, hull_bone_bind_pose_id_b)
            .buf_arg(CreateHullBone_k.Args.hull_inv_bind_pose_indicies, hull_bone_inv_bind_pose_id_b);

        long create_armature_bone_k_ptr = gpu_crud.kernel_ptr(Kernel.create_armature_bone);
        create_armature_bone_k = new CreateArmatureBone_k(GPGPU.sector_cmd_queue_ptr, create_armature_bone_k_ptr)
            .buf_arg(CreateArmatureBone_k.Args.armature_bones, armature_bone_buffer)
            .buf_arg(CreateArmatureBone_k.Args.armature_bone_reference_ids, armature_bone_reference_id_buffer)
            .buf_arg(CreateArmatureBone_k.Args.armature_bone_parent_ids, armature_bone_parent_id_buffer);


        long merge_point_k_ptr = gpu_crud.kernel_ptr(Kernel.merge_point);
        merge_point_k = new MergePoint_k(GPGPU.sector_cmd_queue_ptr, merge_point_k_ptr)
            .buf_arg(MergePoint_k.Args.points_in, point_buffer)
            .buf_arg(MergePoint_k.Args.point_vertex_references_in, point_vertex_reference_buffer)
            .buf_arg(MergePoint_k.Args.point_hull_indices_in, point_hull_index_buffer)
            .buf_arg(MergePoint_k.Args.point_hit_counts_in, point_hit_count_buffer)
            .buf_arg(MergePoint_k.Args.point_flags_in, point_flag_buffer)
            .buf_arg(MergePoint_k.Args.bone_tables_in, point_bone_table_buffer)
            .buf_arg(MergePoint_k.Args.points_out, parent.buffer(BufferType.POINT))
            .buf_arg(MergePoint_k.Args.point_vertex_references_out, parent.buffer(BufferType.POINT_VERTEX_REFERENCE))
            .buf_arg(MergePoint_k.Args.point_hull_indices_out, parent.buffer(BufferType.POINT_HULL_INDEX))
            .buf_arg(MergePoint_k.Args.point_hit_counts_out, parent.buffer(BufferType.POINT_HIT_COUNT))
            .buf_arg(MergePoint_k.Args.point_flags_out, parent.buffer(BufferType.POINT_FLAG))
            .buf_arg(MergePoint_k.Args.bone_tables_out, parent.buffer(BufferType.POINT_BONE_TABLE));

        long merge_edge_k_ptr = gpu_crud.kernel_ptr(Kernel.merge_edge);
        merge_edge_k = new MergeEdge_k(GPGPU.sector_cmd_queue_ptr, merge_edge_k_ptr)
            .buf_arg(MergeEdge_k.Args.edges_in, edge_buffer)
            .buf_arg(MergeEdge_k.Args.edge_lengths_in, edge_length_buffer)
            .buf_arg(MergeEdge_k.Args.edge_flags_in, edge_flag_buffer)
            .buf_arg(MergeEdge_k.Args.edges_out, parent.buffer(BufferType.EDGE))
            .buf_arg(MergeEdge_k.Args.edge_lengths_out, parent.buffer(BufferType.EDGE_LENGTH))
            .buf_arg(MergeEdge_k.Args.edge_flags_out, parent.buffer(BufferType.EDGE_FLAG));

        long merge_hull_bone_k_ptr = gpu_crud.kernel_ptr(Kernel.merge_hull_bone);
        merge_hull_bone_k = new MergeHullBone_k(GPGPU.sector_cmd_queue_ptr, merge_hull_bone_k_ptr)
            .buf_arg(MergeHullBone_k.Args.hull_bones_in, hull_bone_b)
            .buf_arg(MergeHullBone_k.Args.hull_bind_pose_indicies_in, hull_bone_bind_pose_id_b)
            .buf_arg(MergeHullBone_k.Args.hull_inv_bind_pose_indicies_in, hull_bone_inv_bind_pose_id_b)
            .buf_arg(MergeHullBone_k.Args.hull_bones_out, parent.buffer(BufferType.HULL_BONE))
            .buf_arg(MergeHullBone_k.Args.hull_bind_pose_indicies_out, parent.buffer(BufferType.HULL_BONE_BIND_POSE))
            .buf_arg(MergeHullBone_k.Args.hull_inv_bind_pose_indicies_out, parent.buffer(BufferType.HULL_BONE_INV_BIND_POSE));

        long merge_armature_bone_k_ptr = gpu_crud.kernel_ptr(Kernel.merge_armature_bone);
        merge_armature_bone_k = new MergeArmatureBone_k(GPGPU.sector_cmd_queue_ptr, merge_armature_bone_k_ptr)
            .buf_arg(MergeArmatureBone_k.Args.armature_bones_in, armature_bone_buffer)
            .buf_arg(MergeArmatureBone_k.Args.armature_bone_reference_ids_in, armature_bone_reference_id_buffer)
            .buf_arg(MergeArmatureBone_k.Args.armature_bone_parent_ids_in, armature_bone_parent_id_buffer)
            .buf_arg(MergeArmatureBone_k.Args.armature_bones_out, parent.buffer(BufferType.ARMATURE_BONE))
            .buf_arg(MergeArmatureBone_k.Args.armature_bone_reference_ids_out, parent.buffer(BufferType.ARMATURE_BONE_REFERENCE_ID))
            .buf_arg(MergeArmatureBone_k.Args.armature_bone_parent_ids_out, parent.buffer(BufferType.ARMATURE_BONE_PARENT_ID));

        long merge_hull_k_ptr = gpu_crud.kernel_ptr(Kernel.merge_hull);
        merge_hull_k = new MergeHull_k(GPGPU.sector_cmd_queue_ptr, merge_hull_k_ptr)
            .buf_arg(MergeHull_k.Args.hulls_in, hull_b)
            .buf_arg(MergeHull_k.Args.hull_scales_in, hull_scale_b)
            .buf_arg(MergeHull_k.Args.hull_rotations_in, hull_rotation_b)
            .buf_arg(MergeHull_k.Args.hull_frictions_in, hull_friction_b)
            .buf_arg(MergeHull_k.Args.hull_restitutions_in, hull_restitution_b)
            .buf_arg(MergeHull_k.Args.hull_point_tables_in, hull_point_table_b)
            .buf_arg(MergeHull_k.Args.hull_edge_tables_in, hull_edge_table_b)
            .buf_arg(MergeHull_k.Args.bone_tables_in, hull_bone_table_b)
            .buf_arg(MergeHull_k.Args.hull_entity_ids_in, hull_entity_id_b)
            .buf_arg(MergeHull_k.Args.hull_flags_in, hull_flag_b)
            .buf_arg(MergeHull_k.Args.hull_mesh_ids_in, hull_mesh_id_b)
            .buf_arg(MergeHull_k.Args.hull_uv_offsets_in, hull_uv_offset_b)
            .buf_arg(MergeHull_k.Args.hull_integrity_in, hull_integrity_b)
            .buf_arg(MergeHull_k.Args.hulls_out, parent.buffer(BufferType.HULL))
            .buf_arg(MergeHull_k.Args.hull_scales_out, parent.buffer(BufferType.HULL_SCALE))
            .buf_arg(MergeHull_k.Args.hull_rotations_out, parent.buffer(BufferType.HULL_ROTATION))
            .buf_arg(MergeHull_k.Args.hull_frictions_out, parent.buffer(BufferType.HULL_FRICTION))
            .buf_arg(MergeHull_k.Args.hull_restitutions_out, parent.buffer(BufferType.HULL_RESTITUTION))
            .buf_arg(MergeHull_k.Args.hull_point_tables_out, parent.buffer(BufferType.HULL_POINT_TABLE))
            .buf_arg(MergeHull_k.Args.hull_edge_tables_out, parent.buffer(BufferType.HULL_EDGE_TABLE))
            .buf_arg(MergeHull_k.Args.bone_tables_out, parent.buffer(BufferType.HULL_BONE_TABLE))
            .buf_arg(MergeHull_k.Args.hull_entity_ids_out, parent.buffer(BufferType.HULL_ENTITY_ID))
            .buf_arg(MergeHull_k.Args.hull_flags_out, parent.buffer(BufferType.HULL_FLAG))
            .buf_arg(MergeHull_k.Args.hull_mesh_ids_out, parent.buffer(BufferType.HULL_MESH_ID))
            .buf_arg(MergeHull_k.Args.hull_uv_offsets_out, parent.buffer(BufferType.HULL_UV_OFFSET))
            .buf_arg(MergeHull_k.Args.hull_integrity_out, parent.buffer(BufferType.HULL_INTEGRITY));

        long merge_entity_k_ptr = gpu_crud.kernel_ptr(Kernel.merge_entity);
        merge_entity_k = new MergeEntity_k(GPGPU.sector_cmd_queue_ptr, merge_entity_k_ptr)
            .buf_arg(MergeEntity_k.Args.entities_in, entity_buffer)
            .buf_arg(MergeEntity_k.Args.entity_animation_elapsed_in, entity_anim_elapsed_buffer)
            .buf_arg(MergeEntity_k.Args.entity_motion_states_in, entity_motion_state_buffer)
            .buf_arg(MergeEntity_k.Args.entity_animation_indices_in, entity_anim_index_buffer)
            .buf_arg(MergeEntity_k.Args.entity_hull_tables_in, entity_hull_table_buffer)
            .buf_arg(MergeEntity_k.Args.entity_bone_tables_in, entity_bone_table_buffer)
            .buf_arg(MergeEntity_k.Args.entity_masses_in, entity_mass_buffer)
            .buf_arg(MergeEntity_k.Args.entity_root_hulls_in, entity_root_hull_buffer)
            .buf_arg(MergeEntity_k.Args.entity_model_indices_in, entity_model_id_buffer)
            .buf_arg(MergeEntity_k.Args.entity_model_transforms_in, entity_model_transform_buffer)
            .buf_arg(MergeEntity_k.Args.entity_flags_in, entity_flag_buffer)
            .buf_arg(MergeEntity_k.Args.entities_out, parent.buffer(BufferType.ENTITY))
            .buf_arg(MergeEntity_k.Args.entity_animation_elapsed_out, parent.buffer(BufferType.ENTITY_ANIM_ELAPSED))
            .buf_arg(MergeEntity_k.Args.entity_motion_states_out, parent.buffer(BufferType.ENTITY_MOTION_STATE))
            .buf_arg(MergeEntity_k.Args.entity_animation_indices_out, parent.buffer(BufferType.ENTITY_ANIM_INDEX))
            .buf_arg(MergeEntity_k.Args.entity_hull_tables_out, parent.buffer(BufferType.ENTITY_HULL_TABLE))
            .buf_arg(MergeEntity_k.Args.entity_bone_tables_out, parent.buffer(BufferType.ENTITY_BONE_TABLE))
            .buf_arg(MergeEntity_k.Args.entity_masses_out, parent.buffer(BufferType.ENTITY_MASS))
            .buf_arg(MergeEntity_k.Args.entity_root_hulls_out, parent.buffer(BufferType.ENTITY_ROOT_HULL))
            .buf_arg(MergeEntity_k.Args.entity_model_indices_out, parent.buffer(BufferType.ENTITY_MODEL_ID))
            .buf_arg(MergeEntity_k.Args.entity_model_transforms_out, parent.buffer(BufferType.ENTITY_TRANSFORM_ID))
            .buf_arg(MergeEntity_k.Args.entity_flags_out, parent.buffer(BufferType.ENTITY_FLAG));

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
        point_buffer.ensure_capacity(capacity);
        point_anti_gravity_buffer.ensure_capacity(capacity);
        point_vertex_reference_buffer.ensure_capacity(capacity);
        point_hull_index_buffer.ensure_capacity(capacity);
        point_flag_buffer.ensure_capacity(capacity);
        point_hit_count_buffer.ensure_capacity(capacity);
        point_bone_table_buffer.ensure_capacity(capacity);

        var new_point = new float[]{position[0], position[1], position[0], position[1]};
        create_point_k
            .set_arg(CreatePoint_k.Args.target, point_index)
            .set_arg(CreatePoint_k.Args.new_point, new_point)
            .set_arg(CreatePoint_k.Args.new_point_vertex_reference, vertex_index)
            .set_arg(CreatePoint_k.Args.new_point_hull_index, hull_index)
            .set_arg(CreatePoint_k.Args.new_point_hit_count, (short) hit_count)
            .set_arg(CreatePoint_k.Args.new_point_flags, flags)
            .set_arg(CreatePoint_k.Args.new_bone_table, bone_ids)
            .call(GPGPU.global_single_size);

        return point_index++;
    }

    @Override
    public int new_edge(int p1, int p2, float l, int flags)
    {
        int required_capacity = edge_index + 1;
        edge_buffer.ensure_capacity(required_capacity);
        edge_length_buffer.ensure_capacity(required_capacity);
        edge_flag_buffer.ensure_capacity(required_capacity);

        create_edge_k
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
        hull_b.ensure_capacity(capacity);
        hull_scale_b.ensure_capacity(capacity);
        hull_mesh_id_b.ensure_capacity(capacity);
        hull_uv_offset_b.ensure_capacity(capacity);
        hull_rotation_b.ensure_capacity(capacity);
        hull_integrity_b.ensure_capacity(capacity);
        hull_point_table_b.ensure_capacity(capacity);
        hull_edge_table_b.ensure_capacity(capacity);
        hull_flag_b.ensure_capacity(capacity);
        hull_bone_table_b.ensure_capacity(capacity);
        hull_entity_id_b.ensure_capacity(capacity);
        hull_friction_b.ensure_capacity(capacity);
        hull_restitution_b.ensure_capacity(capacity);
        hull_aabb_b.ensure_capacity(capacity);
        hull_aabb_index_b.ensure_capacity(capacity);
        hull_aabb_key_b.ensure_capacity(capacity);

        create_hull_k
            .set_arg(CreateHull_k.Args.target, hull_index)
            .set_arg(CreateHull_k.Args.new_hull, arg_float4(position[0], position[1], position[0], position[1]))
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
    public int new_entity(float x, float y, int[] hull_table, int[] bone_table, float mass, int anim_index, float anim_time, int root_hull, int model_id, int model_transform_id, int flags)
    {
        int capacity = entity_index + 1;
        entity_buffer.ensure_capacity(capacity);
        entity_flag_buffer.ensure_capacity(capacity);
        entity_root_hull_buffer.ensure_capacity(capacity);
        entity_model_id_buffer.ensure_capacity(capacity);
        entity_model_transform_buffer.ensure_capacity(capacity);
        entity_accel_buffer.ensure_capacity(capacity);
        entity_mass_buffer.ensure_capacity(capacity);
        entity_anim_index_buffer.ensure_capacity(capacity);
        entity_anim_elapsed_buffer.ensure_capacity(capacity);
        entity_anim_blend_buffer.ensure_capacity(capacity);
        entity_motion_state_buffer.ensure_capacity(capacity);
        entity_hull_table_buffer.ensure_capacity(capacity);
        entity_bone_table_buffer.ensure_capacity(capacity);

        create_entity_k
            .set_arg(CreateEntity_k.Args.target, entity_index)
            .set_arg(CreateEntity_k.Args.new_entity, arg_float4(x, y, x, y))
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
        hull_bone_b.ensure_capacity(capacity);
        hull_bone_bind_pose_id_b.ensure_capacity(capacity);
        hull_bone_inv_bind_pose_id_b.ensure_capacity(capacity);

        create_bone_k
            .set_arg(CreateHullBone_k.Args.target, hull_bone_index)
            .set_arg(CreateHullBone_k.Args.new_bone, bone_data)
            .set_arg(CreateHullBone_k.Args.new_hull_bind_pose_id, bind_pose_id)
            .set_arg(CreateHullBone_k.Args.new_hull_inv_bind_pose_id, inv_bind_pose_id)
            .call(GPGPU.global_single_size);

        return hull_bone_index++;
    }

    @Override
    public int new_armature_bone(int bone_reference, int bone_parent_id, float[] bone_data)
    {
        int capacity = armature_bone_index + 1;
        armature_bone_buffer.ensure_capacity(capacity);
        armature_bone_reference_id_buffer.ensure_capacity(capacity);
        armature_bone_parent_id_buffer.ensure_capacity(capacity);

        create_armature_bone_k
            .set_arg(CreateArmatureBone_k.Args.target, armature_bone_index)
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone, bone_data)
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone_reference, bone_reference)
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone_parent_id, bone_parent_id)
            .call(GPGPU.global_single_size);

        return armature_bone_index++;
    }

    @Override
    public void destroy()
    {
        gpu_crud.destroy();

        edge_buffer.release();
        edge_length_buffer.release();
        edge_flag_buffer.release();
        hull_b.release();
        hull_scale_b.release();
        hull_mesh_id_b.release();
        hull_uv_offset_b.release();
        hull_rotation_b.release();
        hull_integrity_b.release();
        hull_point_table_b.release();
        hull_edge_table_b.release();
        hull_flag_b.release();
        hull_bone_table_b.release();
        hull_entity_id_b.release();
        hull_aabb_b.release();
        hull_aabb_index_b.release();
        hull_aabb_key_b.release();
        hull_bone_b.release();
        hull_bone_bind_pose_id_b.release();
        hull_bone_inv_bind_pose_id_b.release();
        hull_friction_b.release();
        hull_restitution_b.release();
        point_buffer.release();
        point_anti_gravity_buffer.release();
        point_vertex_reference_buffer.release();
        point_hull_index_buffer.release();
        point_flag_buffer.release();
        point_hit_count_buffer.release();
        point_bone_table_buffer.release();
        bone_reference_buffer.release();
        bone_bind_pose_buffer.release();
        bone_anim_channel_table_buffer.release();
        armature_bone_buffer.release();
        armature_bone_reference_id_buffer.release();
        armature_bone_parent_id_buffer.release();
        entity_buffer.release();
        entity_flag_buffer.release();
        entity_root_hull_buffer.release();
        entity_model_id_buffer.release();
        entity_model_transform_buffer.release();
        entity_accel_buffer.release();
        entity_mass_buffer.release();
        entity_anim_index_buffer.release();
        entity_anim_elapsed_buffer.release();
        entity_anim_blend_buffer.release();
        entity_motion_state_buffer.release();
        entity_hull_table_buffer.release();
        entity_bone_table_buffer.release();

        debug();
    }

    private void debug()
    {
        long total = 0;
        total += edge_buffer.debug_data();
        total += edge_length_buffer.debug_data();
        total += edge_flag_buffer.debug_data();
        total += hull_b.debug_data();
        total += hull_scale_b.debug_data();
        total += hull_mesh_id_b.debug_data();
        total += hull_uv_offset_b.debug_data();
        total += hull_rotation_b.debug_data();
        total += hull_integrity_b.debug_data();
        total += hull_point_table_b.debug_data();
        total += hull_edge_table_b.debug_data();
        total += hull_flag_b.debug_data();
        total += hull_bone_table_b.debug_data();
        total += hull_entity_id_b.debug_data();
        total += hull_aabb_b.debug_data();
        total += hull_aabb_index_b.debug_data();
        total += hull_aabb_key_b.debug_data();
        total += hull_bone_b.debug_data();
        total += hull_bone_bind_pose_id_b.debug_data();
        total += hull_bone_inv_bind_pose_id_b.debug_data();
        total += hull_friction_b.debug_data();
        total += hull_restitution_b.debug_data();
        total += point_buffer.debug_data();
        total += point_anti_gravity_buffer.debug_data();
        total += point_vertex_reference_buffer.debug_data();
        total += point_hull_index_buffer.debug_data();
        total += point_flag_buffer.debug_data();
        total += point_hit_count_buffer.debug_data();
        total += point_bone_table_buffer.debug_data();
        total += bone_reference_buffer.debug_data();
        total += bone_bind_pose_buffer.debug_data();
        total += bone_anim_channel_table_buffer.debug_data();
        total += armature_bone_buffer.debug_data();
        total += armature_bone_reference_id_buffer.debug_data();
        total += armature_bone_parent_id_buffer.debug_data();
        total += entity_buffer.debug_data();
        total += entity_flag_buffer.debug_data();
        total += entity_root_hull_buffer.debug_data();
        total += entity_model_id_buffer.debug_data();
        total += entity_model_transform_buffer.debug_data();
        total += entity_accel_buffer.debug_data();
        total += entity_mass_buffer.debug_data();
        total += entity_anim_index_buffer.debug_data();
        total += entity_anim_elapsed_buffer.debug_data();
        total += entity_anim_blend_buffer.debug_data();
        total += entity_motion_state_buffer.debug_data();
        total += entity_hull_table_buffer.debug_data();
        total += entity_bone_table_buffer.debug_data();

        //System.out.println("---------------------------");
        System.out.println("Sector Memory Usage: MB " + ((float) total / 1024f / 1024f));
    }

    @Override
    public void merge_into_parent()
    {
        merge_point_k
                .set_arg(MergePoint_k.Args.hull_offset, parent.next_hull())
                .set_arg(MergePoint_k.Args.point_offset, parent.next_point())
                .set_arg(MergePoint_k.Args.bone_offset, parent.next_hull_bone())
                .call(arg_long(point_index));

        merge_edge_k
                .set_arg(MergeEdge_k.Args.edge_offset, parent.next_edge())
                .set_arg(MergeEdge_k.Args.point_offset, parent.next_point())
                .call(arg_long(edge_index));

        merge_hull_k
                .set_arg(MergeHull_k.Args.hull_offset, parent.next_hull())
                .set_arg(MergeHull_k.Args.hull_bone_offset, parent.next_hull_bone())
                .set_arg(MergeHull_k.Args.point_offset, parent.next_point())
                .set_arg(MergeHull_k.Args.edge_offset, parent.next_edge())
                .set_arg(MergeHull_k.Args.entity_offset, parent.next_entity())
                .call(arg_long(hull_index));

        merge_entity_k
                .set_arg(MergeEntity_k.Args.entity_offset, parent.next_entity())
                .set_arg(MergeEntity_k.Args.hull_offset, parent.next_hull())
                .set_arg(MergeEntity_k.Args.armature_bone_offset, parent.next_armature_bone())
                .call(arg_long(entity_index));

        merge_hull_bone_k
                .set_arg(MergeHullBone_k.Args.hull_bone_offset, parent.next_hull_bone())
                .set_arg(MergeHullBone_k.Args.armature_bone_offset, parent.next_armature_bone())
                .call(arg_long(hull_bone_index));

        merge_armature_bone_k
                .set_arg(MergeArmatureBone_k.Args.armature_bone_offset, parent.next_armature_bone())
                .call(arg_long(armature_bone_index));
    }
}
