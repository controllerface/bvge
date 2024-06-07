package com.controllerface.bvge.ecs.systems.sectors;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.buffers.BufferType;
import com.controllerface.bvge.cl.buffers.PersistentBuffer;
import com.controllerface.bvge.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.*;
import static com.controllerface.bvge.cl.CLUtils.arg_short2;

public class SectorInputBuffer implements SectorContainer
{
    private final GPUProgram p_gpu_crud = new GPUCrud();

    private final GPUKernel k_create_point;
    private final GPUKernel k_create_edge;
    private final GPUKernel k_create_hull;
    private final GPUKernel k_create_entity;
    private final GPUKernel k_create_hull_bone;
    private final GPUKernel k_create_armature_bone;

    private final GPUKernel k_merge_point;
    private final GPUKernel k_merge_edge;
    private final GPUKernel k_merge_hull;
    private final GPUKernel k_merge_entity;
    private final GPUKernel k_merge_hull_bone;
    private final GPUKernel k_merge_armature_bone;

    //#region Armature Buffers

    private final ResizableBuffer b_armature_bone;
    private final ResizableBuffer b_armature_bone_reference_id;
    private final ResizableBuffer b_armature_bone_parent_id;

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

    private int point_index           = 0;
    private int edge_index            = 0;
    private int hull_index            = 0;
    private int entity_index          = 0;
    private int hull_bone_index       = 0;
    private int armature_bone_index   = 0;

    private final long ptr_queue;
    private final long ptr_egress_sizes;


    public SectorInputBuffer(long ptr_queue, GPUCoreMemory core_memory)
    {
        this.ptr_queue         = ptr_queue;
        this.ptr_egress_sizes  = GPGPU.cl_new_pinned_buffer(CLSize.cl_int * 7);

        // persistent buffers
        b_entity_anim_elapsed        = new PersistentBuffer(this.ptr_queue, CLSize.cl_float2, 1_000L);
        b_entity_motion_state        = new PersistentBuffer(this.ptr_queue, CLSize.cl_short2, 1_000L);
        b_entity_anim_index          = new PersistentBuffer(this.ptr_queue, CLSize.cl_int2, 1_000L);
        b_armature_bone              = new PersistentBuffer(this.ptr_queue, CLSize.cl_float16);
        b_armature_bone_reference_id = new PersistentBuffer(this.ptr_queue, CLSize.cl_int);
        b_armature_bone_parent_id    = new PersistentBuffer(this.ptr_queue, CLSize.cl_int);
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

        long k_ptr_create_point = p_gpu_crud.kernel_ptr(Kernel.create_point);
        k_create_point = new CreatePoint_k(this.ptr_queue, k_ptr_create_point)
            .buf_arg(CreatePoint_k.Args.points, b_point)
            .buf_arg(CreatePoint_k.Args.point_vertex_references, b_point_vertex_reference)
            .buf_arg(CreatePoint_k.Args.point_hull_indices, b_point_hull_index)
            .buf_arg(CreatePoint_k.Args.point_hit_counts, b_point_hit_count)
            .buf_arg(CreatePoint_k.Args.point_flags, b_point_flag)
            .buf_arg(CreatePoint_k.Args.bone_tables, b_point_bone_table);

        long k_ptr_create_edge = p_gpu_crud.kernel_ptr(Kernel.create_edge);
        k_create_edge = new CreateEdge_k(this.ptr_queue, k_ptr_create_edge)
            .buf_arg(CreateEdge_k.Args.edges, b_edge)
            .buf_arg(CreateEdge_k.Args.edge_lengths, b_edge_length)
            .buf_arg(CreateEdge_k.Args.edge_flags, b_edge_flag);

        long k_ptr_create_hull = p_gpu_crud.kernel_ptr(Kernel.create_hull);
        k_create_hull = new CreateHull_k(this.ptr_queue, k_ptr_create_hull)
            .buf_arg(CreateHull_k.Args.hulls, b_hull)
            .buf_arg(CreateHull_k.Args.hull_scales, b_hull_scale)
            .buf_arg(CreateHull_k.Args.hull_rotations, b_hull_rotation)
            .buf_arg(CreateHull_k.Args.hull_frictions, b_hull_friction)
            .buf_arg(CreateHull_k.Args.hull_restitutions, b_hull_restitution)
            .buf_arg(CreateHull_k.Args.hull_point_tables, b_hull_point_table)
            .buf_arg(CreateHull_k.Args.hull_edge_tables, b_hull_edge_table)
            .buf_arg(CreateHull_k.Args.hull_bone_tables, b_hull_bone_table)
            .buf_arg(CreateHull_k.Args.hull_entity_ids, b_hull_entity_id)
            .buf_arg(CreateHull_k.Args.hull_flags, b_hull_flag)
            .buf_arg(CreateHull_k.Args.hull_mesh_ids, b_hull_mesh_id)
            .buf_arg(CreateHull_k.Args.hull_uv_offsets, b_hull_uv_offset)
            .buf_arg(CreateHull_k.Args.hull_integrity, b_hull_integrity);

        long k_ptr_create_entity = p_gpu_crud.kernel_ptr(Kernel.create_entity);
        k_create_entity = new CreateEntity_k(this.ptr_queue, k_ptr_create_entity)
            .buf_arg(CreateEntity_k.Args.entities, b_entity)
            .buf_arg(CreateEntity_k.Args.entity_root_hulls, b_entity_root_hull)
            .buf_arg(CreateEntity_k.Args.entity_model_indices, b_entity_model_id)
            .buf_arg(CreateEntity_k.Args.entity_model_transforms, b_entity_model_transform)
            .buf_arg(CreateEntity_k.Args.entity_flags, b_entity_flag)
            .buf_arg(CreateEntity_k.Args.entity_hull_tables, b_entity_hull_table)
            .buf_arg(CreateEntity_k.Args.entity_bone_tables, b_entity_bone_table)
            .buf_arg(CreateEntity_k.Args.entity_masses, b_entity_mass)
            .buf_arg(CreateEntity_k.Args.entity_animation_indices, b_entity_anim_index)
            .buf_arg(CreateEntity_k.Args.entity_animation_elapsed, b_entity_anim_elapsed)
            .buf_arg(CreateEntity_k.Args.entity_motion_states, b_entity_motion_state);

        long k_ptr_create_hull_bone = p_gpu_crud.kernel_ptr(Kernel.create_hull_bone);
        k_create_hull_bone = new CreateHullBone_k(this.ptr_queue, k_ptr_create_hull_bone)
            .buf_arg(CreateHullBone_k.Args.bones, b_hull_bone)
            .buf_arg(CreateHullBone_k.Args.hull_bind_pose_indicies, b_hull_bone_bind_pose_id)
            .buf_arg(CreateHullBone_k.Args.hull_inv_bind_pose_indicies, b_hull_bone_inv_bind_pose_id);

        long k_ptr_create_armature_bone = p_gpu_crud.kernel_ptr(Kernel.create_armature_bone);
        k_create_armature_bone = new CreateArmatureBone_k(this.ptr_queue, k_ptr_create_armature_bone)
            .buf_arg(CreateArmatureBone_k.Args.armature_bones, b_armature_bone)
            .buf_arg(CreateArmatureBone_k.Args.armature_bone_reference_ids, b_armature_bone_reference_id)
            .buf_arg(CreateArmatureBone_k.Args.armature_bone_parent_ids, b_armature_bone_parent_id);

        long k_ptr_merge_point = p_gpu_crud.kernel_ptr(Kernel.merge_point);
        k_merge_point = new MergePoint_k(this.ptr_queue, k_ptr_merge_point)
            .buf_arg(MergePoint_k.Args.points_in, b_point)
            .buf_arg(MergePoint_k.Args.point_vertex_references_in, b_point_vertex_reference)
            .buf_arg(MergePoint_k.Args.point_hull_indices_in, b_point_hull_index)
            .buf_arg(MergePoint_k.Args.point_hit_counts_in, b_point_hit_count)
            .buf_arg(MergePoint_k.Args.point_flags_in, b_point_flag)
            .buf_arg(MergePoint_k.Args.point_bone_tables_in, b_point_bone_table)
            .buf_arg(MergePoint_k.Args.points_out, core_memory.buffer(BufferType.POINT))
            .buf_arg(MergePoint_k.Args.point_vertex_references_out, core_memory.buffer(BufferType.POINT_VERTEX_REFERENCE))
            .buf_arg(MergePoint_k.Args.point_hull_indices_out, core_memory.buffer(BufferType.POINT_HULL_INDEX))
            .buf_arg(MergePoint_k.Args.point_hit_counts_out, core_memory.buffer(BufferType.POINT_HIT_COUNT))
            .buf_arg(MergePoint_k.Args.point_flags_out, core_memory.buffer(BufferType.POINT_FLAG))
            .buf_arg(MergePoint_k.Args.point_bone_tables_out, core_memory.buffer(BufferType.POINT_BONE_TABLE));

        long k_ptr_merge_edge = p_gpu_crud.kernel_ptr(Kernel.merge_edge);
        k_merge_edge = new MergeEdge_k(this.ptr_queue, k_ptr_merge_edge)
            .buf_arg(MergeEdge_k.Args.edges_in, b_edge)
            .buf_arg(MergeEdge_k.Args.edge_lengths_in, b_edge_length)
            .buf_arg(MergeEdge_k.Args.edge_flags_in, b_edge_flag)
            .buf_arg(MergeEdge_k.Args.edges_out, core_memory.buffer(BufferType.EDGE))
            .buf_arg(MergeEdge_k.Args.edge_lengths_out, core_memory.buffer(BufferType.EDGE_LENGTH))
            .buf_arg(MergeEdge_k.Args.edge_flags_out, core_memory.buffer(BufferType.EDGE_FLAG));

        long k_ptr_merge_hull = p_gpu_crud.kernel_ptr(Kernel.merge_hull);
        k_merge_hull = new MergeHull_k(this.ptr_queue, k_ptr_merge_hull)
            .buf_arg(MergeHull_k.Args.hulls_in, b_hull)
            .buf_arg(MergeHull_k.Args.hull_scales_in, b_hull_scale)
            .buf_arg(MergeHull_k.Args.hull_rotations_in, b_hull_rotation)
            .buf_arg(MergeHull_k.Args.hull_frictions_in, b_hull_friction)
            .buf_arg(MergeHull_k.Args.hull_restitutions_in, b_hull_restitution)
            .buf_arg(MergeHull_k.Args.hull_point_tables_in, b_hull_point_table)
            .buf_arg(MergeHull_k.Args.hull_edge_tables_in, b_hull_edge_table)
            .buf_arg(MergeHull_k.Args.bone_tables_in, b_hull_bone_table)
            .buf_arg(MergeHull_k.Args.hull_entity_ids_in, b_hull_entity_id)
            .buf_arg(MergeHull_k.Args.hull_flags_in, b_hull_flag)
            .buf_arg(MergeHull_k.Args.hull_mesh_ids_in, b_hull_mesh_id)
            .buf_arg(MergeHull_k.Args.hull_uv_offsets_in, b_hull_uv_offset)
            .buf_arg(MergeHull_k.Args.hull_integrity_in, b_hull_integrity)
            .buf_arg(MergeHull_k.Args.hulls_out, core_memory.buffer(BufferType.HULL))
            .buf_arg(MergeHull_k.Args.hull_scales_out, core_memory.buffer(BufferType.HULL_SCALE))
            .buf_arg(MergeHull_k.Args.hull_rotations_out, core_memory.buffer(BufferType.HULL_ROTATION))
            .buf_arg(MergeHull_k.Args.hull_frictions_out, core_memory.buffer(BufferType.HULL_FRICTION))
            .buf_arg(MergeHull_k.Args.hull_restitutions_out, core_memory.buffer(BufferType.HULL_RESTITUTION))
            .buf_arg(MergeHull_k.Args.hull_point_tables_out, core_memory.buffer(BufferType.HULL_POINT_TABLE))
            .buf_arg(MergeHull_k.Args.hull_edge_tables_out, core_memory.buffer(BufferType.HULL_EDGE_TABLE))
            .buf_arg(MergeHull_k.Args.bone_tables_out, core_memory.buffer(BufferType.HULL_BONE_TABLE))
            .buf_arg(MergeHull_k.Args.hull_entity_ids_out, core_memory.buffer(BufferType.HULL_ENTITY_ID))
            .buf_arg(MergeHull_k.Args.hull_flags_out, core_memory.buffer(BufferType.HULL_FLAG))
            .buf_arg(MergeHull_k.Args.hull_mesh_ids_out, core_memory.buffer(BufferType.HULL_MESH_ID))
            .buf_arg(MergeHull_k.Args.hull_uv_offsets_out, core_memory.buffer(BufferType.HULL_UV_OFFSET))
            .buf_arg(MergeHull_k.Args.hull_integrity_out, core_memory.buffer(BufferType.HULL_INTEGRITY));

        long k_ptr_merge_entity = p_gpu_crud.kernel_ptr(Kernel.merge_entity);
        k_merge_entity = new MergeEntity_k(this.ptr_queue, k_ptr_merge_entity)
            .buf_arg(MergeEntity_k.Args.entities_in, b_entity)
            .buf_arg(MergeEntity_k.Args.entity_animation_elapsed_in, b_entity_anim_elapsed)
            .buf_arg(MergeEntity_k.Args.entity_motion_states_in, b_entity_motion_state)
            .buf_arg(MergeEntity_k.Args.entity_animation_indices_in, b_entity_anim_index)
            .buf_arg(MergeEntity_k.Args.entity_hull_tables_in, b_entity_hull_table)
            .buf_arg(MergeEntity_k.Args.entity_bone_tables_in, b_entity_bone_table)
            .buf_arg(MergeEntity_k.Args.entity_masses_in, b_entity_mass)
            .buf_arg(MergeEntity_k.Args.entity_root_hulls_in, b_entity_root_hull)
            .buf_arg(MergeEntity_k.Args.entity_model_indices_in, b_entity_model_id)
            .buf_arg(MergeEntity_k.Args.entity_model_transforms_in, b_entity_model_transform)
            .buf_arg(MergeEntity_k.Args.entity_flags_in, b_entity_flag)
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
            .buf_arg(MergeHullBone_k.Args.hull_bones_in, b_hull_bone)
            .buf_arg(MergeHullBone_k.Args.hull_bind_pose_indicies_in, b_hull_bone_bind_pose_id)
            .buf_arg(MergeHullBone_k.Args.hull_inv_bind_pose_indicies_in, b_hull_bone_inv_bind_pose_id)
            .buf_arg(MergeHullBone_k.Args.hull_bones_out, core_memory.buffer(BufferType.HULL_BONE))
            .buf_arg(MergeHullBone_k.Args.hull_bind_pose_indicies_out, core_memory.buffer(BufferType.HULL_BONE_BIND_POSE))
            .buf_arg(MergeHullBone_k.Args.hull_inv_bind_pose_indicies_out, core_memory.buffer(BufferType.HULL_BONE_INV_BIND_POSE));

        long k_ptr_merge_armature_bone = p_gpu_crud.kernel_ptr(Kernel.merge_armature_bone);
        k_merge_armature_bone = new MergeArmatureBone_k(this.ptr_queue, k_ptr_merge_armature_bone)
            .buf_arg(MergeArmatureBone_k.Args.armature_bones_in, b_armature_bone)
            .buf_arg(MergeArmatureBone_k.Args.armature_bone_reference_ids_in, b_armature_bone_reference_id)
            .buf_arg(MergeArmatureBone_k.Args.armature_bone_parent_ids_in, b_armature_bone_parent_id)
            .buf_arg(MergeArmatureBone_k.Args.armature_bones_out, core_memory.buffer(BufferType.ARMATURE_BONE))
            .buf_arg(MergeArmatureBone_k.Args.armature_bone_reference_ids_out, core_memory.buffer(BufferType.ARMATURE_BONE_REFERENCE_ID))
            .buf_arg(MergeArmatureBone_k.Args.armature_bone_parent_ids_out, core_memory.buffer(BufferType.ARMATURE_BONE_PARENT_ID));
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
        b_point.ensure_capacity(capacity);
        b_point_vertex_reference.ensure_capacity(capacity);
        b_point_hull_index.ensure_capacity(capacity);
        b_point_flag.ensure_capacity(capacity);
        b_point_hit_count.ensure_capacity(capacity);
        b_point_bone_table.ensure_capacity(capacity);

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
            .set_arg(CreatePoint_k.Args.new_bone_table, bone_ids)
            .call(GPGPU.global_single_size);

        return point_index++;
    }

    @Override
    public int new_edge(int p1, int p2, float l, int flags)
    {
        int required_capacity = edge_index + 1;
        b_edge.ensure_capacity(required_capacity);
        b_edge_length.ensure_capacity(required_capacity);
        b_edge_flag.ensure_capacity(required_capacity);

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
        b_hull.ensure_capacity(capacity);
        b_hull_scale.ensure_capacity(capacity);
        b_hull_mesh_id.ensure_capacity(capacity);
        b_hull_uv_offset.ensure_capacity(capacity);
        b_hull_rotation.ensure_capacity(capacity);
        b_hull_integrity.ensure_capacity(capacity);
        b_hull_point_table.ensure_capacity(capacity);
        b_hull_edge_table.ensure_capacity(capacity);
        b_hull_flag.ensure_capacity(capacity);
        b_hull_bone_table.ensure_capacity(capacity);
        b_hull_entity_id.ensure_capacity(capacity);
        b_hull_friction.ensure_capacity(capacity);
        b_hull_restitution.ensure_capacity(capacity);

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
        b_entity.ensure_capacity(capacity);
        b_entity_flag.ensure_capacity(capacity);
        b_entity_root_hull.ensure_capacity(capacity);
        b_entity_model_id.ensure_capacity(capacity);
        b_entity_model_transform.ensure_capacity(capacity);
        b_entity_mass.ensure_capacity(capacity);
        b_entity_anim_index.ensure_capacity(capacity);
        b_entity_anim_elapsed.ensure_capacity(capacity);
        b_entity_motion_state.ensure_capacity(capacity);
        b_entity_hull_table.ensure_capacity(capacity);
        b_entity_bone_table.ensure_capacity(capacity);

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
        b_hull_bone.ensure_capacity(capacity);
        b_hull_bone_bind_pose_id.ensure_capacity(capacity);
        b_hull_bone_inv_bind_pose_id.ensure_capacity(capacity);

        k_create_hull_bone
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
        b_armature_bone.ensure_capacity(capacity);
        b_armature_bone_reference_id.ensure_capacity(capacity);
        b_armature_bone_parent_id.ensure_capacity(capacity);

        k_create_armature_bone
            .set_arg(CreateArmatureBone_k.Args.target, armature_bone_index)
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone, bone_data)
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone_reference, bone_reference)
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone_parent_id, bone_parent_id)
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

        if (armature_bone_index > 0) k_merge_armature_bone
            .set_arg(MergeArmatureBone_k.Args.armature_bone_offset, parent.next_armature_bone())
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
        b_armature_bone.release();
        b_armature_bone_reference_id.release();
        b_armature_bone_parent_id.release();
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
        total += b_armature_bone.debug_data();
        total += b_armature_bone_reference_id.debug_data();
        total += b_armature_bone_parent_id.debug_data();
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
        System.out.println("World Buffer Memory Usage: MB " + ((float) total / 1024f / 1024f));
    }
}
