package com.controllerface.bvge.ecs.systems.sectors;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.cl.buffers.BufferGroup;
import com.controllerface.bvge.cl.buffers.BufferType;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.*;
import static com.controllerface.bvge.cl.CLUtils.arg_short2;

public class OrderedSectorInput
{
    private final long ptr_queue;
    private final GPUProgram p_gpu_crud;
    private final BufferGroup sector_group;

    private final GPUKernel k_create_point;
    private final GPUKernel k_create_edge;
    private final GPUKernel k_create_hull;
    private final GPUKernel k_create_entity;
    private final GPUKernel k_create_hull_bone;
    private final GPUKernel k_create_entity_bone;

    private int point_index           = 0;
    private int edge_index            = 0;
    private int hull_index            = 0;
    private int entity_index          = 0;
    private int hull_bone_index       = 0;
    private int entity_bone_index     = 0;

    public OrderedSectorInput(long ptrQueue, GPUProgram pGpuCrud, BufferGroup sectorGroup)
    {
        this.ptr_queue = ptrQueue;
        this.p_gpu_crud = pGpuCrud;
        sector_group = sectorGroup;

        long k_ptr_create_point = this.p_gpu_crud.kernel_ptr(Kernel.create_point);
        k_create_point = new CreatePoint_k(this.ptr_queue, k_ptr_create_point)
            .buf_arg(CreatePoint_k.Args.points, sector_group.buffer(BufferType.POINT))
            .buf_arg(CreatePoint_k.Args.point_vertex_references, sector_group.buffer(BufferType.POINT_VERTEX_REFERENCE))
            .buf_arg(CreatePoint_k.Args.point_hull_indices, sector_group.buffer(BufferType.POINT_HULL_INDEX))
            .buf_arg(CreatePoint_k.Args.point_hit_counts, sector_group.buffer(BufferType.POINT_HIT_COUNT))
            .buf_arg(CreatePoint_k.Args.point_flags, sector_group.buffer(BufferType.POINT_FLAG))
            .buf_arg(CreatePoint_k.Args.point_bone_tables, sector_group.buffer(BufferType.POINT_BONE_TABLE));

        long k_ptr_create_edge = this.p_gpu_crud.kernel_ptr(Kernel.create_edge);
        k_create_edge = new CreateEdge_k(this.ptr_queue, k_ptr_create_edge)
            .buf_arg(CreateEdge_k.Args.edges, sector_group.buffer(BufferType.EDGE))
            .buf_arg(CreateEdge_k.Args.edge_lengths, sector_group.buffer(BufferType.EDGE_LENGTH))
            .buf_arg(CreateEdge_k.Args.edge_flags, sector_group.buffer(BufferType.EDGE_FLAG));

        long k_ptr_create_hull = this.p_gpu_crud.kernel_ptr(Kernel.create_hull);
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

        long k_ptr_create_entity = this.p_gpu_crud.kernel_ptr(Kernel.create_entity);
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

        long k_ptr_create_hull_bone = this.p_gpu_crud.kernel_ptr(Kernel.create_hull_bone);
        k_create_hull_bone = new CreateHullBone_k(this.ptr_queue, k_ptr_create_hull_bone)
            .buf_arg(CreateHullBone_k.Args.hull_bones, sector_group.buffer(BufferType.HULL_BONE))
            .buf_arg(CreateHullBone_k.Args.hull_bind_pose_indicies, sector_group.buffer(BufferType.HULL_BONE_BIND_POSE))
            .buf_arg(CreateHullBone_k.Args.hull_inv_bind_pose_indicies, sector_group.buffer(BufferType.HULL_BONE_INV_BIND_POSE));

        long k_ptr_create_entity_bone = this.p_gpu_crud.kernel_ptr(Kernel.create_entity_bone);
        k_create_entity_bone = new CreateEntityBone_k(this.ptr_queue, k_ptr_create_entity_bone)
            .buf_arg(CreateEntityBone_k.Args.entity_bones, sector_group.buffer(BufferType.ENTITY_BONE))
            .buf_arg(CreateEntityBone_k.Args.entity_bone_reference_ids, sector_group.buffer(BufferType.ENTITY_BONE_REFERENCE_ID))
            .buf_arg(CreateEntityBone_k.Args.entity_bone_parent_ids, sector_group.buffer(BufferType.ENTITY_BONE_PARENT_ID));
    }

    public int point_index()
    {
        return point_index;
    }

    public int edge_index()
    {
        return edge_index;
    }

    public int hull_index()
    {
        return hull_index;
    }

    public int entity_index()
    {
        return entity_index;
    }

    public int hull_bone_index()
    {
        return hull_bone_index;
    }

    public int entity_bone_index()
    {
        return entity_bone_index;
    }

    public int create_point(float[] position, int[] bone_ids, int vertex_index, int hull_index, int hit_count, int flags)
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

    public int create_edge(int p1, int p2, float l, int flags)
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

    public int create_hull(int mesh_id, float[] position, float[] scale, float[] rotation, int[] point_table, int[] edge_table, int[] bone_table, float friction, float restitution, int entity_id, int uv_offset, int flags)
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

    public int create_entity(float x, float y, float z, float w, int[] hull_table, int[] bone_table, float mass, int anim_index, float anim_time, int root_hull, int model_id, int model_transform_id, int flags)
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

    public int create_hull_bone(float[] bone_data, int bind_pose_id, int inv_bind_pose_id)
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

    public int create_armature_bone(int bone_reference, int bone_parent_id, float[] bone_data)
    {
        int capacity = entity_bone_index + 1;
        sector_group.buffer(BufferType.ENTITY_BONE).ensure_capacity(capacity);
        sector_group.buffer(BufferType.ENTITY_BONE_REFERENCE_ID).ensure_capacity(capacity);
        sector_group.buffer(BufferType.ENTITY_BONE_PARENT_ID).ensure_capacity(capacity);

        k_create_entity_bone
            .set_arg(CreateEntityBone_k.Args.target, entity_bone_index)
            .set_arg(CreateEntityBone_k.Args.new_armature_bone, bone_data)
            .set_arg(CreateEntityBone_k.Args.new_armature_bone_reference, bone_reference)
            .set_arg(CreateEntityBone_k.Args.new_armature_bone_parent_id, bone_parent_id)
            .call(GPGPU.global_single_size);

        return entity_bone_index++;
    }

    public void reset()
    {
        point_index           = 0;
        edge_index            = 0;
        hull_index            = 0;
        entity_index          = 0;
        hull_bone_index       = 0;
        entity_bone_index     = 0;
    }
}
