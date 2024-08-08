package com.controllerface.bvge.memory.sectors;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.GPUResource;
import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.buffers.TransientBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.egress.*;
import com.controllerface.bvge.gpu.cl.programs.crud.GPUCrud;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.groups.UnorderedCoreBufferGroup;

import static com.controllerface.bvge.gpu.GPU.CL.arg_long;
import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.cl_int;
import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class UnorderedSectorOutput implements GPUResource
{
    private static final long ENTITY_INIT = 1_000L;
    private static final long HULL_INIT   = 1_000L;
    private static final long EDGE_INIT   = 2_400L;
    private static final long POINT_INIT  = 5_000L;

    private final GPUProgram p_gpu_crud;
    private final GPUKernel k_egress_entities;
    private final GPUKernel k_egress_hulls;
    private final GPUKernel k_egress_edges;
    private final GPUKernel k_egress_points;
    private final GPUKernel k_egress_hull_bones;
    private final GPUKernel k_egress_entity_bones;
    private final CL_CommandQueue cmd_queue;
    private final CL_Buffer egress_sizes_buf;
    private final UnorderedCoreBufferGroup sector_buffers;
    private final GPUCoreMemory core_memory;

    private final ResizableBuffer b_entity_bone_shift;
    private final ResizableBuffer b_hull_bone_shift;
    private final ResizableBuffer b_edge_shift;
    private final ResizableBuffer b_hull_shift;
    private final ResizableBuffer b_point_shift;

    public UnorderedSectorOutput(String name,
                                 CL_CommandQueue cmd_queue,
                                 GPUCoreMemory core_memory,
                                 long entity_init,
                                 long hull_init,
                                 long edge_init,
                                 long point_init)
    {
        this.cmd_queue = cmd_queue;
        this.core_memory       = core_memory;
        this.egress_sizes_buf  = GPU.CL.new_pinned_buffer(GPGPU.compute.context, (long)cl_int.size() * 6);
        this.sector_buffers    = new UnorderedCoreBufferGroup(name, this.cmd_queue, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        this.p_gpu_crud        = new GPUCrud().init();

        b_hull_shift                 = new TransientBuffer(cmd_queue, cl_int.size(), hull_init);
        b_edge_shift                 = new TransientBuffer(cmd_queue, cl_int.size(), edge_init);
        b_point_shift                = new TransientBuffer(cmd_queue, cl_int.size(), point_init);
        b_hull_bone_shift            = new TransientBuffer(cmd_queue, cl_int.size(), hull_init);
        b_entity_bone_shift          = new TransientBuffer(cmd_queue, cl_int.size(), entity_init);

        k_egress_entities = new EgressEntities_k(this.cmd_queue, this.p_gpu_crud)
            .buf_arg(EgressEntities_k.Args.point_hull_indices_in,           this.core_memory.get_buffer(POINT_HULL_INDEX))
            .buf_arg(EgressEntities_k.Args.point_bone_tables_in,            this.core_memory.get_buffer(POINT_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.edges_in,                        this.core_memory.get_buffer(EDGE))
            .buf_arg(EgressEntities_k.Args.edge_pins_in,                    this.core_memory.get_buffer(EDGE_PIN))
            .buf_arg(EgressEntities_k.Args.hull_point_tables_in,            this.core_memory.get_buffer(HULL_POINT_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_edge_tables_in,             this.core_memory.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_bone_tables_in,             this.core_memory.get_buffer(HULL_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_bind_pose_indices_in,       this.core_memory.get_buffer(HULL_BONE_BIND_POSE))
            .buf_arg(EgressEntities_k.Args.entity_bone_parent_ids_in,       this.core_memory.get_buffer(ENTITY_BONE_PARENT_ID))
            .buf_arg(EgressEntities_k.Args.entities_in,                     this.core_memory.get_buffer(ENTITY))
            .buf_arg(EgressEntities_k.Args.entity_animation_time_in,        this.core_memory.get_buffer(ENTITY_ANIM_TIME))
            .buf_arg(EgressEntities_k.Args.entity_previous_time_in,         this.core_memory.get_buffer(ENTITY_PREV_TIME))
            .buf_arg(EgressEntities_k.Args.entity_motion_states_in,         this.core_memory.get_buffer(ENTITY_MOTION_STATE))
            .buf_arg(EgressEntities_k.Args.entity_animation_layers_in,      this.core_memory.get_buffer(ENTITY_ANIM_LAYER))
            .buf_arg(EgressEntities_k.Args.entity_previous_layers_in,       this.core_memory.get_buffer(ENTITY_PREV_LAYER))
            .buf_arg(EgressEntities_k.Args.entity_hull_tables_in,           this.core_memory.get_buffer(ENTITY_HULL_TABLE))
            .buf_arg(EgressEntities_k.Args.entity_bone_tables_in,           this.core_memory.get_buffer(ENTITY_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.entity_masses_in,                this.core_memory.get_buffer(ENTITY_MASS))
            .buf_arg(EgressEntities_k.Args.entity_root_hulls_in,            this.core_memory.get_buffer(ENTITY_ROOT_HULL))
            .buf_arg(EgressEntities_k.Args.entity_model_indices_in,         this.core_memory.get_buffer(ENTITY_MODEL_ID))
            .buf_arg(EgressEntities_k.Args.entity_model_transforms_in,      this.core_memory.get_buffer(ENTITY_TRANSFORM_ID))
            .buf_arg(EgressEntities_k.Args.entity_types_in,                 this.core_memory.get_buffer(ENTITY_TYPE))
            .buf_arg(EgressEntities_k.Args.entity_flags_in,                 this.core_memory.get_buffer(ENTITY_FLAG))
            .buf_arg(EgressEntities_k.Args.entities_out,                    sector_buffers.buffer(ENTITY))
            .buf_arg(EgressEntities_k.Args.entity_animation_time_out,       sector_buffers.buffer(ENTITY_ANIM_TIME))
            .buf_arg(EgressEntities_k.Args.entity_previous_time_out,        sector_buffers.buffer(ENTITY_PREV_TIME))
            .buf_arg(EgressEntities_k.Args.entity_motion_states_out,        sector_buffers.buffer(ENTITY_MOTION_STATE))
            .buf_arg(EgressEntities_k.Args.entity_animation_layers_out,     sector_buffers.buffer(ENTITY_ANIM_LAYER))
            .buf_arg(EgressEntities_k.Args.entity_previous_layers_out,      sector_buffers.buffer(ENTITY_PREV_LAYER))
            .buf_arg(EgressEntities_k.Args.entity_hull_tables_out,          sector_buffers.buffer(ENTITY_HULL_TABLE))
            .buf_arg(EgressEntities_k.Args.entity_bone_tables_out,          sector_buffers.buffer(ENTITY_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.entity_masses_out,               sector_buffers.buffer(ENTITY_MASS))
            .buf_arg(EgressEntities_k.Args.entity_root_hulls_out,           sector_buffers.buffer(ENTITY_ROOT_HULL))
            .buf_arg(EgressEntities_k.Args.entity_model_indices_out,        sector_buffers.buffer(ENTITY_MODEL_ID))
            .buf_arg(EgressEntities_k.Args.entity_model_transforms_out,     sector_buffers.buffer(ENTITY_TRANSFORM_ID))
            .buf_arg(EgressEntities_k.Args.entity_types_out,                sector_buffers.buffer(ENTITY_TYPE))
            .buf_arg(EgressEntities_k.Args.entity_flags_out,                sector_buffers.buffer(ENTITY_FLAG))
            .buf_arg(EgressEntities_k.Args.new_points,                      b_point_shift)
            .buf_arg(EgressEntities_k.Args.new_edges,                       b_edge_shift)
            .buf_arg(EgressEntities_k.Args.new_hulls,                       b_hull_shift)
            .buf_arg(EgressEntities_k.Args.new_hull_bones,                  b_hull_bone_shift)
            .buf_arg(EgressEntities_k.Args.new_entity_bones,                b_entity_bone_shift)
            .buf_arg(EgressEntities_k.Args.counters,                        egress_sizes_buf);

        k_egress_hulls = new EgressHulls_k(this.cmd_queue, this.p_gpu_crud)
            .buf_arg(EgressHulls_k.Args.hulls_in,                        this.core_memory.get_buffer(HULL))
            .buf_arg(EgressHulls_k.Args.hull_scales_in,                  this.core_memory.get_buffer(HULL_SCALE))
            .buf_arg(EgressHulls_k.Args.hull_rotations_in,               this.core_memory.get_buffer(HULL_ROTATION))
            .buf_arg(EgressHulls_k.Args.hull_frictions_in,               this.core_memory.get_buffer(HULL_FRICTION))
            .buf_arg(EgressHulls_k.Args.hull_restitutions_in,            this.core_memory.get_buffer(HULL_RESTITUTION))
            .buf_arg(EgressHulls_k.Args.hull_point_tables_in,            this.core_memory.get_buffer(HULL_POINT_TABLE))
            .buf_arg(EgressHulls_k.Args.hull_edge_tables_in,             this.core_memory.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(EgressHulls_k.Args.hull_bone_tables_in,             this.core_memory.get_buffer(HULL_BONE_TABLE))
            .buf_arg(EgressHulls_k.Args.hull_entity_ids_in,              this.core_memory.get_buffer(HULL_ENTITY_ID))
            .buf_arg(EgressHulls_k.Args.hull_flags_in,                   this.core_memory.get_buffer(HULL_FLAG))
            .buf_arg(EgressHulls_k.Args.hull_mesh_ids_in,                this.core_memory.get_buffer(HULL_MESH_ID))
            .buf_arg(EgressHulls_k.Args.hull_uv_offsets_in,              this.core_memory.get_buffer(HULL_UV_OFFSET))
            .buf_arg(EgressHulls_k.Args.hull_integrity_in,               this.core_memory.get_buffer(HULL_INTEGRITY))
            .buf_arg(EgressHulls_k.Args.hulls_out,                       sector_buffers.buffer(HULL))
            .buf_arg(EgressHulls_k.Args.hull_scales_out,                 sector_buffers.buffer(HULL_SCALE))
            .buf_arg(EgressHulls_k.Args.hull_rotations_out,              sector_buffers.buffer(HULL_ROTATION))
            .buf_arg(EgressHulls_k.Args.hull_frictions_out,              sector_buffers.buffer(HULL_FRICTION))
            .buf_arg(EgressHulls_k.Args.hull_restitutions_out,           sector_buffers.buffer(HULL_RESTITUTION))
            .buf_arg(EgressHulls_k.Args.hull_point_tables_out,           sector_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(EgressHulls_k.Args.hull_edge_tables_out,            sector_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(EgressHulls_k.Args.hull_bone_tables_out,            sector_buffers.buffer(HULL_BONE_TABLE))
            .buf_arg(EgressHulls_k.Args.hull_entity_ids_out,             sector_buffers.buffer(HULL_ENTITY_ID))
            .buf_arg(EgressHulls_k.Args.hull_flags_out,                  sector_buffers.buffer(HULL_FLAG))
            .buf_arg(EgressHulls_k.Args.hull_mesh_ids_out,               sector_buffers.buffer(HULL_MESH_ID))
            .buf_arg(EgressHulls_k.Args.hull_uv_offsets_out,             sector_buffers.buffer(HULL_UV_OFFSET))
            .buf_arg(EgressHulls_k.Args.hull_integrity_out,              sector_buffers.buffer(HULL_INTEGRITY))
            .buf_arg(EgressHulls_k.Args.new_hulls,                       b_hull_shift);

        k_egress_edges = new EgressEdges_k(this.cmd_queue, this.p_gpu_crud)
            .buf_arg(EgressEdges_k.Args.edges_in,                        this.core_memory.get_buffer(EDGE))
            .buf_arg(EgressEdges_k.Args.edge_lengths_in,                 this.core_memory.get_buffer(EDGE_LENGTH))
            .buf_arg(EgressEdges_k.Args.edge_flags_in,                   this.core_memory.get_buffer(EDGE_FLAG))
            .buf_arg(EgressEdges_k.Args.edge_pins_in,                    this.core_memory.get_buffer(EDGE_PIN))
            .buf_arg(EgressEdges_k.Args.edges_out,                       sector_buffers.buffer(EDGE))
            .buf_arg(EgressEdges_k.Args.edge_lengths_out,                sector_buffers.buffer(EDGE_LENGTH))
            .buf_arg(EgressEdges_k.Args.edge_flags_out,                  sector_buffers.buffer(EDGE_FLAG))
            .buf_arg(EgressEdges_k.Args.edge_pins_out,                   sector_buffers.buffer(EDGE_PIN))
            .buf_arg(EgressEdges_k.Args.new_edges,                       b_edge_shift);

        k_egress_points = new EgressPoints_k(this.cmd_queue, this.p_gpu_crud)
            .buf_arg(EgressPoints_k.Args.points_in,                       this.core_memory.get_buffer(POINT))
            .buf_arg(EgressPoints_k.Args.point_vertex_references_in,      this.core_memory.get_buffer(POINT_VERTEX_REFERENCE))
            .buf_arg(EgressPoints_k.Args.point_hull_indices_in,           this.core_memory.get_buffer(POINT_HULL_INDEX))
            .buf_arg(EgressPoints_k.Args.point_hit_counts_in,             this.core_memory.get_buffer(POINT_HIT_COUNT))
            .buf_arg(EgressPoints_k.Args.point_flags_in,                  this.core_memory.get_buffer(POINT_FLAG))
            .buf_arg(EgressPoints_k.Args.point_bone_tables_in,            this.core_memory.get_buffer(POINT_BONE_TABLE))
            .buf_arg(EgressPoints_k.Args.points_out,                      sector_buffers.buffer(POINT))
            .buf_arg(EgressPoints_k.Args.point_vertex_references_out,     sector_buffers.buffer(POINT_VERTEX_REFERENCE))
            .buf_arg(EgressPoints_k.Args.point_hull_indices_out,          sector_buffers.buffer(POINT_HULL_INDEX))
            .buf_arg(EgressPoints_k.Args.point_hit_counts_out,            sector_buffers.buffer(POINT_HIT_COUNT))
            .buf_arg(EgressPoints_k.Args.point_flags_out,                 sector_buffers.buffer(POINT_FLAG))
            .buf_arg(EgressPoints_k.Args.point_bone_tables_out,           sector_buffers.buffer(POINT_BONE_TABLE))
            .buf_arg(EgressPoints_k.Args.new_points,                      b_point_shift);

        k_egress_hull_bones = new EgressHullBones_k(this.cmd_queue, this.p_gpu_crud)
            .buf_arg(EgressHullBones_k.Args.hull_bones_in,                   this.core_memory.get_buffer(HULL_BONE))
            .buf_arg(EgressHullBones_k.Args.hull_bind_pose_indicies_in,      this.core_memory.get_buffer(HULL_BONE_BIND_POSE))
            .buf_arg(EgressHullBones_k.Args.hull_inv_bind_pose_indicies_in,  this.core_memory.get_buffer(HULL_BONE_INV_BIND_POSE))
            .buf_arg(EgressHullBones_k.Args.hull_bones_out,                  sector_buffers.buffer(HULL_BONE))
            .buf_arg(EgressHullBones_k.Args.hull_bind_pose_indicies_out,     sector_buffers.buffer(HULL_BONE_BIND_POSE))
            .buf_arg(EgressHullBones_k.Args.hull_inv_bind_pose_indicies_out, sector_buffers.buffer(HULL_BONE_INV_BIND_POSE))
            .buf_arg(EgressHullBones_k.Args.new_hull_bones,                  b_hull_bone_shift);

        k_egress_entity_bones = new EgressEntityBones_k(this.cmd_queue, this.p_gpu_crud)
            .buf_arg(EgressEntityBones_k.Args.entity_bones_in,               this.core_memory.get_buffer(ENTITY_BONE))
            .buf_arg(EgressEntityBones_k.Args.entity_bone_reference_ids_in,  this.core_memory.get_buffer(ENTITY_BONE_REFERENCE_ID))
            .buf_arg(EgressEntityBones_k.Args.entity_bone_parent_ids_in,     this.core_memory.get_buffer(ENTITY_BONE_PARENT_ID))
            .buf_arg(EgressEntityBones_k.Args.entity_bones_out,              sector_buffers.buffer(ENTITY_BONE))
            .buf_arg(EgressEntityBones_k.Args.entity_bone_reference_ids_out, sector_buffers.buffer(ENTITY_BONE_REFERENCE_ID))
            .buf_arg(EgressEntityBones_k.Args.entity_bone_parent_ids_out,    sector_buffers.buffer(ENTITY_BONE_PARENT_ID))
            .buf_arg(EgressEntityBones_k.Args.new_entity_bones,              b_entity_bone_shift);
    }

    public void egress(int entity_count, int[] egress_counts)
    {
        GPU.CL.zero_buffer(cmd_queue, egress_sizes_buf, (long)cl_int.size() * 6);
        int entity_capacity        = egress_counts[0];
        int hull_capacity          = egress_counts[1];
        int point_capacity         = egress_counts[2];
        int edge_capacity          = egress_counts[3];
        int hull_bone_capacity     = egress_counts[4];
        int entity_bone_capacity   = egress_counts[5];

        int entity_size = GPGPU.compute.calculate_preferred_global_size(entity_count);

        int hull_count        = core_memory.sector_container().next_hull();
        int edge_count        = core_memory.sector_container().next_edge();
        int point_count       = core_memory.sector_container().next_point();
        int hull_bone_count   = core_memory.sector_container().next_hull_bone();
        int entity_bone_count = core_memory.sector_container().next_entity_bone();

        int hull_size        = GPGPU.compute.calculate_preferred_global_size(hull_count);
        int edge_size        = GPGPU.compute.calculate_preferred_global_size(edge_count);
        int point_size       = GPGPU.compute.calculate_preferred_global_size(point_count);
        int hull_bone_size   = GPGPU.compute.calculate_preferred_global_size(hull_bone_count);
        int entity_bone_size = GPGPU.compute.calculate_preferred_global_size(entity_bone_count);

        b_hull_shift.ensure_capacity(hull_count);
        b_edge_shift.ensure_capacity(edge_count);
        b_point_shift.ensure_capacity(point_count);
        b_hull_bone_shift.ensure_capacity(hull_bone_count);
        b_entity_bone_shift.ensure_capacity(entity_bone_count);

        b_hull_shift.clear_negative();
        b_edge_shift.clear_negative();
        b_point_shift.clear_negative();
        b_hull_bone_shift.clear_negative();
        b_entity_bone_shift.clear_negative();

        sector_buffers.ensure_capacity_all(point_capacity, edge_capacity, hull_capacity, entity_capacity, hull_bone_capacity, entity_bone_capacity);
        k_egress_entities
            .set_arg(EgressEntities_k.Args.max_entity, entity_count)
            .call(arg_long(entity_size), GPGPU.compute.preferred_work_size);

        k_egress_hulls
            .set_arg(EgressHulls_k.Args.max_hull, hull_count)
            .call(arg_long(hull_size), GPGPU.compute.preferred_work_size);

        k_egress_edges
            .set_arg(EgressEdges_k.Args.max_edge, edge_count)
            .call(arg_long(edge_size), GPGPU.compute.preferred_work_size);

        k_egress_points
            .set_arg(EgressPoints_k.Args.max_point, point_count)
            .call(arg_long(point_size), GPGPU.compute.preferred_work_size);

        k_egress_hull_bones
            .set_arg(EgressHullBones_k.Args.max_hull_bone, hull_bone_count)
            .call(arg_long(hull_bone_size), GPGPU.compute.preferred_work_size);

        k_egress_entity_bones
            .set_arg(EgressEntityBones_k.Args.max_entity_bone, entity_bone_count)
            .call(arg_long(entity_bone_size), GPGPU.compute.preferred_work_size);
    }

    public void unload(UnorderedCoreBufferGroup.Raw raw_sectors, int[] counts)
    {
        sector_buffers.unload_sectors(raw_sectors, counts);
    }

    public void release()
    {
        p_gpu_crud.release();
        sector_buffers.release();
        egress_sizes_buf.release();
    }
}
