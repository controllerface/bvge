package com.controllerface.bvge.memory.sectors;

import com.controllerface.bvge.gpu.GPUResource;
import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.crud.*;
import com.controllerface.bvge.gpu.cl.kernels.egress.CountEgressEntities_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.SectorContainer;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.gpu.GPU.CL.*;
import static com.controllerface.bvge.gpu.cl.CL_DataTypes.*;
import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class SectorController implements SectorContainer, GPUResource
{
    private static final int EGRESS_COUNTERS = 8;
    private static final int EGRESS_COUNTERS_SIZE = cl_int.size() * EGRESS_COUNTERS;

    public static final int ENTITY_INFO_WIDTH = 33;

    private final CoreBufferGroup sector_buffers;

    private final GPUKernel k_create_point;
    private final GPUKernel k_create_edge;
    private final GPUKernel k_create_hull;
    private final GPUKernel k_create_entity;
    private final GPUKernel k_create_hull_bone;
    private final GPUKernel k_create_entity_bone;
    private final GPUKernel k_read_position;
    private final GPUKernel k_read_entity_info;
    private final GPUKernel k_write_entity_info;
    private final GPUKernel k_update_accel;
    private final GPUKernel k_update_select_block;
    private final GPUKernel k_clear_select_block;
    private final GPUKernel k_place_block;
    private final GPUKernel k_update_mouse_position;
    private final GPUKernel k_update_block_position;
    private final GPUKernel k_count_egress_entities;

    private int point_index       = 0;
    private int edge_index        = 0;
    private int hull_index        = 0;
    private int entity_index      = 0;
    private int hull_bone_index   = 0;
    private int entity_bone_index = 0;

    private final long ptr_egress_sizes;
    private final long ptr_position_buffer;
    private final long ptr_info_buffer;
    private final long ptr_queue;

    public SectorController(long ptr_queue, GPUProgram p_gpu_crud, CoreBufferGroup sector_buffers)
    {
        this.ptr_queue = ptr_queue;
        this.sector_buffers = sector_buffers;

        ptr_position_buffer = GPGPU.cl_new_pinned_buffer(cl_float2.size());
        ptr_info_buffer     = GPGPU.cl_new_pinned_buffer((long)cl_float.size() * ENTITY_INFO_WIDTH);
        ptr_egress_sizes    = GPGPU.cl_new_pinned_buffer(EGRESS_COUNTERS_SIZE);

        long k_ptr_create_point = p_gpu_crud.kernel_ptr(KernelType.create_point);
        k_create_point = new CreatePoint_k(this.ptr_queue, k_ptr_create_point)
            .buf_arg(CreatePoint_k.Args.points,                         this.sector_buffers.buffer(POINT))
            .buf_arg(CreatePoint_k.Args.point_vertex_references,        this.sector_buffers.buffer(POINT_VERTEX_REFERENCE))
            .buf_arg(CreatePoint_k.Args.point_hull_indices,             this.sector_buffers.buffer(POINT_HULL_INDEX))
            .buf_arg(CreatePoint_k.Args.point_hit_counts,               this.sector_buffers.buffer(POINT_HIT_COUNT))
            .buf_arg(CreatePoint_k.Args.point_flags,                    this.sector_buffers.buffer(POINT_FLAG))
            .buf_arg(CreatePoint_k.Args.point_bone_tables,              this.sector_buffers.buffer(POINT_BONE_TABLE));

        long k_ptr_create_edge = p_gpu_crud.kernel_ptr(KernelType.create_edge);
        k_create_edge = new CreateEdge_k(this.ptr_queue, k_ptr_create_edge)
            .buf_arg(CreateEdge_k.Args.edges,                           this.sector_buffers.buffer(EDGE))
            .buf_arg(CreateEdge_k.Args.edge_lengths,                    this.sector_buffers.buffer(EDGE_LENGTH))
            .buf_arg(CreateEdge_k.Args.edge_flags,                      this.sector_buffers.buffer(EDGE_FLAG))
            .buf_arg(CreateEdge_k.Args.edge_pins,                       this.sector_buffers.buffer(EDGE_PIN));

        long k_ptr_create_hull = p_gpu_crud.kernel_ptr(KernelType.create_hull);
        k_create_hull = new CreateHull_k(this.ptr_queue, k_ptr_create_hull)
            .buf_arg(CreateHull_k.Args.hulls,                           this.sector_buffers.buffer(HULL))
            .buf_arg(CreateHull_k.Args.hull_scales,                     this.sector_buffers.buffer(HULL_SCALE))
            .buf_arg(CreateHull_k.Args.hull_rotations,                  this.sector_buffers.buffer(HULL_ROTATION))
            .buf_arg(CreateHull_k.Args.hull_frictions,                  this.sector_buffers.buffer(HULL_FRICTION))
            .buf_arg(CreateHull_k.Args.hull_restitutions,               this.sector_buffers.buffer(HULL_RESTITUTION))
            .buf_arg(CreateHull_k.Args.hull_point_tables,               this.sector_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(CreateHull_k.Args.hull_edge_tables,                this.sector_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(CreateHull_k.Args.hull_bone_tables,                this.sector_buffers.buffer(HULL_BONE_TABLE))
            .buf_arg(CreateHull_k.Args.hull_entity_ids,                 this.sector_buffers.buffer(HULL_ENTITY_ID))
            .buf_arg(CreateHull_k.Args.hull_flags,                      this.sector_buffers.buffer(HULL_FLAG))
            .buf_arg(CreateHull_k.Args.hull_mesh_ids,                   this.sector_buffers.buffer(HULL_MESH_ID))
            .buf_arg(CreateHull_k.Args.hull_uv_offsets,                 this.sector_buffers.buffer(HULL_UV_OFFSET))
            .buf_arg(CreateHull_k.Args.hull_integrity,                  this.sector_buffers.buffer(HULL_INTEGRITY));

        long k_ptr_create_entity = p_gpu_crud.kernel_ptr(KernelType.create_entity);
        k_create_entity = new CreateEntity_k(this.ptr_queue, k_ptr_create_entity)
            .buf_arg(CreateEntity_k.Args.entities,                      this.sector_buffers.buffer(ENTITY))
            .buf_arg(CreateEntity_k.Args.entity_root_hulls,             this.sector_buffers.buffer(ENTITY_ROOT_HULL))
            .buf_arg(CreateEntity_k.Args.entity_model_indices,          this.sector_buffers.buffer(ENTITY_MODEL_ID))
            .buf_arg(CreateEntity_k.Args.entity_model_transforms,       this.sector_buffers.buffer(ENTITY_TRANSFORM_ID))
            .buf_arg(CreateEntity_k.Args.entity_types,                  this.sector_buffers.buffer(ENTITY_TYPE))
            .buf_arg(CreateEntity_k.Args.entity_flags,                  this.sector_buffers.buffer(ENTITY_FLAG))
            .buf_arg(CreateEntity_k.Args.entity_hull_tables,            this.sector_buffers.buffer(ENTITY_HULL_TABLE))
            .buf_arg(CreateEntity_k.Args.entity_bone_tables,            this.sector_buffers.buffer(ENTITY_BONE_TABLE))
            .buf_arg(CreateEntity_k.Args.entity_masses,                 this.sector_buffers.buffer(ENTITY_MASS))
            .buf_arg(CreateEntity_k.Args.entity_animation_layers,       this.sector_buffers.buffer(ENTITY_ANIM_LAYER))
            .buf_arg(CreateEntity_k.Args.entity_previous_layers,        this.sector_buffers.buffer(ENTITY_PREV_LAYER))
            .buf_arg(CreateEntity_k.Args.entity_animation_time,         this.sector_buffers.buffer(ENTITY_ANIM_TIME))
            .buf_arg(CreateEntity_k.Args.entity_previous_time,          this.sector_buffers.buffer(ENTITY_PREV_TIME))
            .buf_arg(CreateEntity_k.Args.entity_motion_states,          this.sector_buffers.buffer(ENTITY_MOTION_STATE));

        long k_ptr_create_hull_bone = p_gpu_crud.kernel_ptr(KernelType.create_hull_bone);
        k_create_hull_bone = new CreateHullBone_k(this.ptr_queue, k_ptr_create_hull_bone)
            .buf_arg(CreateHullBone_k.Args.hull_bones,                  this.sector_buffers.buffer(HULL_BONE))
            .buf_arg(CreateHullBone_k.Args.hull_bind_pose_indicies,     this.sector_buffers.buffer(HULL_BONE_BIND_POSE))
            .buf_arg(CreateHullBone_k.Args.hull_inv_bind_pose_indicies, this.sector_buffers.buffer(HULL_BONE_INV_BIND_POSE));

        long k_ptr_create_entity_bone = p_gpu_crud.kernel_ptr(KernelType.create_entity_bone);
        k_create_entity_bone = new CreateEntityBone_k(this.ptr_queue, k_ptr_create_entity_bone)
            .buf_arg(CreateEntityBone_k.Args.entity_bones,              this.sector_buffers.buffer(ENTITY_BONE))
            .buf_arg(CreateEntityBone_k.Args.entity_bone_reference_ids, this.sector_buffers.buffer(ENTITY_BONE_REFERENCE_ID))
            .buf_arg(CreateEntityBone_k.Args.entity_bone_parent_ids,    this.sector_buffers.buffer(ENTITY_BONE_PARENT_ID));

        long k_ptr_read_position = p_gpu_crud.kernel_ptr(KernelType.read_position);
        k_read_position = new ReadPosition_k(this.ptr_queue, k_ptr_read_position)
            .buf_arg(ReadPosition_k.Args.entities,                      this.sector_buffers.buffer(ENTITY));

        long k_ptr_read_entity_info = p_gpu_crud.kernel_ptr(KernelType.read_entity_info);
        k_read_entity_info = new ReadEntityInfo_k(this.ptr_queue, k_ptr_read_entity_info)
            .buf_arg(ReadEntityInfo_k.Args.entities,                    this.sector_buffers.buffer(ENTITY))
            .buf_arg(ReadEntityInfo_k.Args.entity_accel,                this.sector_buffers.buffer(ENTITY_ACCEL))
            .buf_arg(ReadEntityInfo_k.Args.entity_motion_states,        this.sector_buffers.buffer(ENTITY_MOTION_STATE))
            .buf_arg(ReadEntityInfo_k.Args.entity_flags,                this.sector_buffers.buffer(ENTITY_FLAG))
            .buf_arg(ReadEntityInfo_k.Args.entity_animation_layers,     this.sector_buffers.buffer(ENTITY_ANIM_LAYER))
            .buf_arg(ReadEntityInfo_k.Args.entity_previous_layers,      this.sector_buffers.buffer(ENTITY_PREV_LAYER))
            .buf_arg(ReadEntityInfo_k.Args.entity_animation_time,       this.sector_buffers.buffer(ENTITY_ANIM_TIME))
            .buf_arg(ReadEntityInfo_k.Args.entity_previous_time,        this.sector_buffers.buffer(ENTITY_PREV_TIME))
            .buf_arg(ReadEntityInfo_k.Args.entity_animation_blend,      this.sector_buffers.buffer(ENTITY_ANIM_BLEND))
            .ptr_arg(ReadEntityInfo_k.Args.output,                      ptr_info_buffer);

        long k_ptr_write_entity_info = p_gpu_crud.kernel_ptr(KernelType.write_entity_info);
        k_write_entity_info = new WriteEntityInfo_k(this.ptr_queue, k_ptr_write_entity_info)
            .buf_arg(WriteEntityInfo_k.Args.entity_accel,               this.sector_buffers.buffer(ENTITY_ACCEL))
            .buf_arg(WriteEntityInfo_k.Args.entity_animation_time,      this.sector_buffers.buffer(ENTITY_ANIM_TIME))
            .buf_arg(WriteEntityInfo_k.Args.entity_previous_time,       this.sector_buffers.buffer(ENTITY_PREV_TIME))
            .buf_arg(WriteEntityInfo_k.Args.entity_animation_blend,     this.sector_buffers.buffer(ENTITY_ANIM_BLEND))
            .buf_arg(WriteEntityInfo_k.Args.entity_motion_states,       this.sector_buffers.buffer(ENTITY_MOTION_STATE))
            .buf_arg(WriteEntityInfo_k.Args.entity_animation_layers,    this.sector_buffers.buffer(ENTITY_ANIM_LAYER))
            .buf_arg(WriteEntityInfo_k.Args.entity_previous_layers,     this.sector_buffers.buffer(ENTITY_PREV_LAYER))
            .buf_arg(WriteEntityInfo_k.Args.entity_flags,               this.sector_buffers.buffer(ENTITY_FLAG));


        long k_ptr_update_accel = p_gpu_crud.kernel_ptr(KernelType.update_accel);
        k_update_accel = new UpdateAccel_k(this.ptr_queue, k_ptr_update_accel)
            .buf_arg(UpdateAccel_k.Args.entity_accel,                   this.sector_buffers.buffer(ENTITY_ACCEL));

        long k_ptr_update_mouse_position = p_gpu_crud.kernel_ptr(KernelType.update_mouse_position);
        k_update_mouse_position = new UpdateMousePosition_k(this.ptr_queue, k_ptr_update_mouse_position)
            .buf_arg(UpdateMousePosition_k.Args.entity_root_hulls,      this.sector_buffers.buffer(ENTITY_ROOT_HULL))
            .buf_arg(UpdateMousePosition_k.Args.hull_point_tables,      this.sector_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(UpdateMousePosition_k.Args.points,                 this.sector_buffers.buffer(POINT));

        long k_ptr_update_block_position = p_gpu_crud.kernel_ptr(KernelType.update_block_position);
        k_update_block_position = new UpdateBlockPosition_k(this.ptr_queue, k_ptr_update_block_position)
            .buf_arg(UpdateBlockPosition_k.Args.entities,               this.sector_buffers.buffer(ENTITY))
            .buf_arg(UpdateBlockPosition_k.Args.entity_root_hulls,      this.sector_buffers.buffer(ENTITY_ROOT_HULL))
            .buf_arg(UpdateBlockPosition_k.Args.hull_point_tables,      this.sector_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(UpdateBlockPosition_k.Args.points,                 this.sector_buffers.buffer(POINT));

        long k_ptr_count_egress_candidates = p_gpu_crud.kernel_ptr(KernelType.count_egress_entities);
        k_count_egress_entities = new CountEgressEntities_k(this.ptr_queue, k_ptr_count_egress_candidates)
            .buf_arg(CountEgressEntities_k.Args.entity_flags,           this.sector_buffers.buffer(ENTITY_FLAG))
            .buf_arg(CountEgressEntities_k.Args.entity_hull_tables,     this.sector_buffers.buffer(ENTITY_HULL_TABLE))
            .buf_arg(CountEgressEntities_k.Args.entity_bone_tables,     this.sector_buffers.buffer(ENTITY_BONE_TABLE))
            .buf_arg(CountEgressEntities_k.Args.hull_flags,             this.sector_buffers.buffer(HULL_FLAG))
            .buf_arg(CountEgressEntities_k.Args.hull_point_tables,      this.sector_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(CountEgressEntities_k.Args.hull_edge_tables,       this.sector_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(CountEgressEntities_k.Args.hull_bone_tables,       this.sector_buffers.buffer(HULL_BONE_TABLE))
            .ptr_arg(CountEgressEntities_k.Args.counters,               ptr_egress_sizes);


        long k_ptr_update_select_block = p_gpu_crud.kernel_ptr(KernelType.update_select_block);
        k_update_select_block = new UpdateSelectBlock_k(this.ptr_queue, k_ptr_update_select_block)
            .buf_arg(UpdateSelectBlock_k.Args.entity_flags,             this.sector_buffers.buffer(ENTITY_FLAG))
            .buf_arg(UpdateSelectBlock_k.Args.hull_uv_offsets,          this.sector_buffers.buffer(HULL_UV_OFFSET))
            .buf_arg(UpdateSelectBlock_k.Args.entity_hull_tables,       this.sector_buffers.buffer(ENTITY_HULL_TABLE));

        long k_ptr_clear_select_block = p_gpu_crud.kernel_ptr(KernelType.clear_select_block);
        k_clear_select_block = new ClearSelectBlock_k(this.ptr_queue, k_ptr_clear_select_block)
            .buf_arg(ClearSelectBlock_k.Args.entity_flags,              this.sector_buffers.buffer(ENTITY_FLAG));

        long k_ptr_place_block = p_gpu_crud.kernel_ptr(KernelType.place_block);
        k_place_block = new PlaceBlock_k(this.ptr_queue, k_ptr_place_block)
                .buf_arg(PlaceBlock_k.Args.entities,                    this.sector_buffers.buffer(ENTITY))
                .buf_arg(PlaceBlock_k.Args.entity_hull_tables,          this.sector_buffers.buffer(ENTITY_HULL_TABLE))
                .buf_arg(PlaceBlock_k.Args.hulls,                       this.sector_buffers.buffer(HULL))
                .buf_arg(PlaceBlock_k.Args.hull_point_tables,           this.sector_buffers.buffer(HULL_POINT_TABLE))
                .buf_arg(PlaceBlock_k.Args.hull_rotations,              this.sector_buffers.buffer(HULL_ROTATION))
                .buf_arg(PlaceBlock_k.Args.points,                      this.sector_buffers.buffer(POINT));
    }

    public void reset()
    {
        point_index       = 0;
        edge_index        = 0;
        hull_index        = 0;
        entity_index      = 0;
        hull_bone_index   = 0;
        entity_bone_index = 0;
    }

    public void expand(int point_count,
                       int edge_count,
                       int hull_count,
                       int entity_count,
                       int hull_bone_count,
                       int armature_bone_count)
    {
        point_index       += point_count;
        edge_index        += edge_count;
        hull_index        += hull_count;
        entity_index      += entity_count;
        hull_bone_index   += hull_bone_count;
        entity_bone_index += armature_bone_count;
    }

    public void compact(int[] shift_counts) // todo: unify ordering of counts in all places, should be: point, edge, hull, entity, hull bone, entity bone
    {
        edge_index        -= (shift_counts[0]);
        hull_bone_index   -= (shift_counts[1]);
        point_index       -= (shift_counts[2]);
        hull_index        -= (shift_counts[3]);
        entity_index      -= (shift_counts[4]);
        entity_bone_index -= (shift_counts[5]);
    }

    public void update_accel(int entity_index, float acc_x, float acc_y)
    {
        k_update_accel
            .set_arg(UpdateAccel_k.Args.target, entity_index)
            .set_arg(UpdateAccel_k.Args.new_value, arg_float2(acc_x, acc_y))
            .call_task();
    }

    public float[] read_position(int entity_index)
    {
        GPGPU.cl_zero_buffer(this.ptr_queue, ptr_position_buffer, cl_float2.size());

        k_read_position
            .ptr_arg(ReadPosition_k.Args.output, ptr_position_buffer)
            .set_arg(ReadPosition_k.Args.target, entity_index)
            .call_task();

        return GPGPU.cl_read_pinned_float_buffer(this.ptr_queue, ptr_position_buffer, cl_float.size(), 2);
    }

    public void read_entity_info(int entity_index, float[] output)
    {
        GPGPU.cl_zero_buffer(this.ptr_queue, ptr_info_buffer, cl_float2.size());

        k_read_entity_info
            .ptr_arg(ReadEntityInfo_k.Args.output, ptr_info_buffer)
            .set_arg(ReadEntityInfo_k.Args.target, entity_index)
            .call_task();

        GPGPU.cl_read_pinned_float_buffer(this.ptr_queue, ptr_info_buffer, cl_float.size(), ENTITY_INFO_WIDTH, output);
    }

    public void write_entity_info(int target,
                                  float[] accel,
                                  float[] current_time,
                                  float[] previous_time,
                                  float[] current_blend,
                                  short[] motion_state,
                                  int[] anim_layers,
                                  int[] anim_previous,
                                  int arm_flag)
    {
        k_write_entity_info
            .set_arg(WriteEntityInfo_k.Args.target,            target)
            .set_arg(WriteEntityInfo_k.Args.new_accel,         accel)
            .set_arg(WriteEntityInfo_k.Args.new_anim_time,     current_time)
            .set_arg(WriteEntityInfo_k.Args.new_prev_time,     previous_time)
            .set_arg(WriteEntityInfo_k.Args.new_anim_blend,    current_blend)
            .set_arg(WriteEntityInfo_k.Args.new_motion_state,  motion_state)
            .set_arg(WriteEntityInfo_k.Args.new_anim_layers,   anim_layers)
            .set_arg(WriteEntityInfo_k.Args.new_anim_previous, anim_previous)
            .set_arg(WriteEntityInfo_k.Args.new_flags,         arm_flag)
            .call_task();
    }

    public int[] count_egress_entities()
    {
        GPGPU.cl_zero_buffer(this.ptr_queue, ptr_egress_sizes, EGRESS_COUNTERS_SIZE);
        int entity_count = next_entity();
        int entity_size  = GPGPU.calculate_preferred_global_size(entity_count);
        k_count_egress_entities
            .set_arg(CountEgressEntities_k.Args.max_entity, entity_count)
            .call(arg_long(entity_size), GPGPU.preferred_work_size);
        return GPGPU.cl_read_pinned_int_buffer(this.ptr_queue, ptr_egress_sizes, cl_int.size(), EGRESS_COUNTERS);
    }

    public void update_mouse_position(int entity_index, float x, float y)
    {
        k_update_mouse_position
            .set_arg(UpdateMousePosition_k.Args.target, entity_index)
            .set_arg(UpdateMousePosition_k.Args.new_value, arg_float2(x, y))
            .call_task();
    }

    public void update_block_position(int entity_index, float x, float y)
    {
        k_update_block_position
                .set_arg(UpdateBlockPosition_k.Args.target, entity_index)
                .set_arg(UpdateBlockPosition_k.Args.new_value, arg_float2(x, y))
                .call_task();
    }

    public void update_block_cursor(int entity_index, int new_uv)
    {
        k_update_select_block
            .set_arg(UpdateSelectBlock_k.Args.target, entity_index)
            .set_arg(UpdateSelectBlock_k.Args.new_value, new_uv)
            .call_task();
    }

    public void clear_block_cursor(int entity_index)
    {
        k_clear_select_block
            .set_arg(ClearSelectBlock_k.Args.target, entity_index)
            .call_task();
    }

    public void place_block(int src, int dest)
    {
        k_place_block
            .set_arg(PlaceBlock_k.Args.src, src)
            .set_arg(PlaceBlock_k.Args.dest, dest)
            .call_task();
    }

    @Override
    public int create_point(float[] position,
                            int[] bone_ids,
                            int vertex_index,
                            int hull_index,
                            int hit_count,
                            int flags)
    {
        int capacity = point_index + 1;
        sector_buffers.ensure_point_capacity(capacity);

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
            .call_task();

        return point_index++;
    }

    @Override
    public int create_edge(int p1, int p2, float l, int flags, int edge_pin)
    {
        int capacity = edge_index + 1;
        sector_buffers.ensure_edge_capacity(capacity);

        k_create_edge
            .set_arg(CreateEdge_k.Args.target, edge_index)
            .set_arg(CreateEdge_k.Args.new_edge, arg_int2(p1, p2))
            .set_arg(CreateEdge_k.Args.new_edge_length, l)
            .set_arg(CreateEdge_k.Args.new_edge_flag, flags)
            .set_arg(CreateEdge_k.Args.new_edge_pin, edge_pin)
            .call_task();

        return edge_index++;
    }

    @Override
    public int create_hull(int mesh_id,
                           float[] position,
                           float[] scale,
                           float[] rotation,
                           int[] point_table,
                           int[] edge_table,
                           int[] bone_table,
                           float friction,
                           float restitution,
                           int entity_id,
                           int uv_offset,
                           int flags)
    {
        int capacity = hull_index + 1;
        sector_buffers.ensure_hull_capacity(capacity);

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
            .call_task();

        return hull_index++;
    }

    @Override
    public int create_entity(float x, float y, float z, float w,
                             int[] hull_table, int[] bone_table,
                             float mass,
                             int anim_index,
                             float anim_time,
                             int root_hull,
                             int model_id,
                             int model_transform_id,
                             int type,
                             int flags)
    {
        int capacity = entity_index + 1;
        sector_buffers.ensure_entity_capacity(capacity);

        k_create_entity
            .set_arg(CreateEntity_k.Args.target,                        entity_index)
            .set_arg(CreateEntity_k.Args.new_entity,                    arg_float4(x, y, z, w))
            .set_arg(CreateEntity_k.Args.new_entity_root_hull,          root_hull)
            .set_arg(CreateEntity_k.Args.new_entity_model_id,           model_id)
            .set_arg(CreateEntity_k.Args.new_entity_model_transform,    model_transform_id)
            .set_arg(CreateEntity_k.Args.new_entity_type,               type)
            .set_arg(CreateEntity_k.Args.new_entity_flags,              flags)
            .set_arg(CreateEntity_k.Args.new_entity_hull_table,         hull_table)
            .set_arg(CreateEntity_k.Args.new_entity_bone_table,         bone_table)
            .set_arg(CreateEntity_k.Args.new_entity_mass,               mass)
            .set_arg(CreateEntity_k.Args.new_entity_animation_layer,    arg_int4(anim_index, 0,0,0))
            .set_arg(CreateEntity_k.Args.new_entity_previous_layer,     arg_int4(-1, -1, -1, -1))
            .set_arg(CreateEntity_k.Args.new_entity_animation_time,     arg_float4(anim_time, 0.0f, 0.0f, 0.0f))
            .set_arg(CreateEntity_k.Args.new_entity_previous_time,      arg_float4(0.0f, 0.0f, 0.0f, 0.0f))
            .set_arg(CreateEntity_k.Args.new_entity_animation_state,    arg_short2((short) 0, (short) 0))
            .call_task();

        return entity_index++;
    }

    @Override
    public int create_hull_bone(float[] bone_data, int bind_pose_id, int inv_bind_pose_id)
    {
        int capacity = hull_bone_index + 1;
        sector_buffers.ensure_hull_bone_capacity(capacity);

        k_create_hull_bone
            .set_arg(CreateHullBone_k.Args.target, hull_bone_index)
            .set_arg(CreateHullBone_k.Args.new_hull_bone, bone_data)
            .set_arg(CreateHullBone_k.Args.new_hull_bind_pose_id, bind_pose_id)
            .set_arg(CreateHullBone_k.Args.new_hull_inv_bind_pose_id, inv_bind_pose_id)
            .call_task();

        return hull_bone_index++;
    }

    @Override
    public int create_entity_bone(int bone_reference, int bone_parent_id, float[] bone_data)
    {
        int capacity = entity_bone_index + 1;
        sector_buffers.ensure_entity_bone_capacity(capacity);

        k_create_entity_bone
            .set_arg(CreateEntityBone_k.Args.target, entity_bone_index)
            .set_arg(CreateEntityBone_k.Args.new_armature_bone, bone_data)
            .set_arg(CreateEntityBone_k.Args.new_armature_bone_reference, bone_reference)
            .set_arg(CreateEntityBone_k.Args.new_armature_bone_parent_id, bone_parent_id)
            .call_task();

        return entity_bone_index++;
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
    public int next_entity_bone()
    {
        return entity_bone_index;
    }

    @Override
    public void release()
    {
        GPGPU.cl_release_buffer(ptr_position_buffer);
        GPGPU.cl_release_buffer(ptr_egress_sizes);
    }
}
