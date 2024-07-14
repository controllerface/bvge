package com.controllerface.bvge.game.world.sectors;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.buffers.Destroyable;
import com.controllerface.bvge.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.cl.buffers.TransientBuffer;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.kernels.egress.*;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.GPUProgram;

import static com.controllerface.bvge.cl.CLData.*;
import static com.controllerface.bvge.cl.CLUtils.*;
import static com.controllerface.bvge.cl.buffers.CoreBufferType.*;

public class UnorderedSectorOutput implements Destroyable
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
    private final long ptr_queue;
    private final long ptr_egress_sizes;
    private final UnorderedCoreBufferGroup sector_buffers;
    private final GPUCoreMemory core_memory;

    private final ResizableBuffer b_entity_bone_shift;
    private final ResizableBuffer b_hull_bone_shift;
    private final ResizableBuffer b_edge_shift;
    private final ResizableBuffer b_hull_shift;
    private final ResizableBuffer b_point_shift;

    public UnorderedSectorOutput(String name,
                                 long ptr_queue,
                                 GPUCoreMemory core_memory,
                                 long entity_init,
                                 long hull_init,
                                 long edge_init,
                                 long point_init)
    {
        this.ptr_queue         = ptr_queue;
        this.core_memory       = core_memory;
        this.ptr_egress_sizes  = GPGPU.cl_new_pinned_buffer(cl_int.size() * 6);
        this.sector_buffers    = new UnorderedCoreBufferGroup(name, this.ptr_queue, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        this.p_gpu_crud        = new GPUCrud().init();

        b_hull_shift                 = new TransientBuffer(ptr_queue, cl_int.size(), hull_init);
        b_edge_shift                 = new TransientBuffer(ptr_queue, cl_int.size(), edge_init);
        b_point_shift                = new TransientBuffer(ptr_queue, cl_int.size(), point_init);
        b_hull_bone_shift            = new TransientBuffer(ptr_queue, cl_int.size(), hull_init);
        b_entity_bone_shift          = new TransientBuffer(ptr_queue, cl_int.size(), entity_init);

        long k_ptr_egress_entities = p_gpu_crud.kernel_ptr(Kernel.egress_entities);
        k_egress_entities = new EgressEntities_k(this.ptr_queue, k_ptr_egress_entities)
            .buf_arg(EgressEntities_k.Args.point_hull_indices_in,           this.core_memory.get_buffer(POINT_HULL_INDEX))
            .buf_arg(EgressEntities_k.Args.point_bone_tables_in,            this.core_memory.get_buffer(POINT_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.edges_in,                        this.core_memory.get_buffer(EDGE))
            .buf_arg(EgressEntities_k.Args.hull_point_tables_in,            this.core_memory.get_buffer(HULL_POINT_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_edge_tables_in,             this.core_memory.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_bone_tables_in,             this.core_memory.get_buffer(HULL_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.hull_bind_pose_indices_in,       this.core_memory.get_buffer(HULL_BONE_BIND_POSE))
            .buf_arg(EgressEntities_k.Args.entity_bone_parent_ids_in,       this.core_memory.get_buffer(ENTITY_BONE_PARENT_ID))
            .buf_arg(EgressEntities_k.Args.entities_in,                     this.core_memory.get_buffer(ENTITY))
            .buf_arg(EgressEntities_k.Args.entity_animation_elapsed_in,     this.core_memory.get_buffer(ENTITY_ANIM_ELAPSED))
            .buf_arg(EgressEntities_k.Args.entity_motion_states_in,         this.core_memory.get_buffer(ENTITY_MOTION_STATE))
            .buf_arg(EgressEntities_k.Args.entity_animation_indices_in,     this.core_memory.get_buffer(ENTITY_ANIM_INDEX))
            .buf_arg(EgressEntities_k.Args.entity_hull_tables_in,           this.core_memory.get_buffer(ENTITY_HULL_TABLE))
            .buf_arg(EgressEntities_k.Args.entity_bone_tables_in,           this.core_memory.get_buffer(ENTITY_BONE_TABLE))
            .buf_arg(EgressEntities_k.Args.entity_masses_in,                this.core_memory.get_buffer(ENTITY_MASS))
            .buf_arg(EgressEntities_k.Args.entity_root_hulls_in,            this.core_memory.get_buffer(ENTITY_ROOT_HULL))
            .buf_arg(EgressEntities_k.Args.entity_model_indices_in,         this.core_memory.get_buffer(ENTITY_MODEL_ID))
            .buf_arg(EgressEntities_k.Args.entity_model_transforms_in,      this.core_memory.get_buffer(ENTITY_TRANSFORM_ID))
            .buf_arg(EgressEntities_k.Args.entity_types_in,                 this.core_memory.get_buffer(ENTITY_TYPE))
            .buf_arg(EgressEntities_k.Args.entity_flags_in,                 this.core_memory.get_buffer(ENTITY_FLAG))
            .buf_arg(EgressEntities_k.Args.entities_out,                    sector_buffers.buffer(ENTITY))
            .buf_arg(EgressEntities_k.Args.entity_animation_elapsed_out,    sector_buffers.buffer(ENTITY_ANIM_ELAPSED))
            .buf_arg(EgressEntities_k.Args.entity_motion_states_out,        sector_buffers.buffer(ENTITY_MOTION_STATE))
            .buf_arg(EgressEntities_k.Args.entity_animation_indices_out,    sector_buffers.buffer(ENTITY_ANIM_INDEX))
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
            .ptr_arg(EgressEntities_k.Args.counters,                        ptr_egress_sizes);

        long k_ptr_egress_hulls = p_gpu_crud.kernel_ptr(Kernel.egress_hulls);
        k_egress_hulls = new EgressHulls_k(this.ptr_queue, k_ptr_egress_hulls)
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

        long k_ptr_egress_edges = p_gpu_crud.kernel_ptr(Kernel.egress_edges);
        k_egress_edges = new EgressEdges_k(this.ptr_queue, k_ptr_egress_edges)
            .buf_arg(EgressEdges_k.Args.edges_in,                        this.core_memory.get_buffer(EDGE))
            .buf_arg(EgressEdges_k.Args.edge_lengths_in,                 this.core_memory.get_buffer(EDGE_LENGTH))
            .buf_arg(EgressEdges_k.Args.edge_flags_in,                   this.core_memory.get_buffer(EDGE_FLAG))
            .buf_arg(EgressEdges_k.Args.edges_out,                       sector_buffers.buffer(EDGE))
            .buf_arg(EgressEdges_k.Args.edge_lengths_out,                sector_buffers.buffer(EDGE_LENGTH))
            .buf_arg(EgressEdges_k.Args.edge_flags_out,                  sector_buffers.buffer(EDGE_FLAG))
            .buf_arg(EgressEdges_k.Args.new_edges,                       b_edge_shift);

        long k_ptr_egress_points = p_gpu_crud.kernel_ptr(Kernel.egress_points);
        k_egress_points = new EgressPoints_k(this.ptr_queue, k_ptr_egress_points)
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

        long k_ptr_egress_hull_bones = p_gpu_crud.kernel_ptr(Kernel.egress_hull_bones);
        k_egress_hull_bones = new EgressHullBones_k(this.ptr_queue, k_ptr_egress_hull_bones)
            .buf_arg(EgressHullBones_k.Args.hull_bones_in,                   this.core_memory.get_buffer(HULL_BONE))
            .buf_arg(EgressHullBones_k.Args.hull_bind_pose_indicies_in,      this.core_memory.get_buffer(HULL_BONE_BIND_POSE))
            .buf_arg(EgressHullBones_k.Args.hull_inv_bind_pose_indicies_in,  this.core_memory.get_buffer(HULL_BONE_INV_BIND_POSE))
            .buf_arg(EgressHullBones_k.Args.hull_bones_out,                  sector_buffers.buffer(HULL_BONE))
            .buf_arg(EgressHullBones_k.Args.hull_bind_pose_indicies_out,     sector_buffers.buffer(HULL_BONE_BIND_POSE))
            .buf_arg(EgressHullBones_k.Args.hull_inv_bind_pose_indicies_out, sector_buffers.buffer(HULL_BONE_INV_BIND_POSE))
            .buf_arg(EgressHullBones_k.Args.new_hull_bones,                  b_hull_bone_shift);

        long k_ptr_egress_entity_bones = p_gpu_crud.kernel_ptr(Kernel.egress_entity_bones);
        k_egress_entity_bones = new EgressEntityBones_k(this.ptr_queue, k_ptr_egress_entity_bones)
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
        GPGPU.cl_zero_buffer(ptr_queue, ptr_egress_sizes, cl_int.size() * 6);
        int entity_capacity        = egress_counts[0];
        int hull_capacity          = egress_counts[1];
        int point_capacity         = egress_counts[2];
        int edge_capacity          = egress_counts[3];
        int hull_bone_capacity     = egress_counts[4];
        int entity_bone_capacity   = egress_counts[5];

        int entity_size = GPGPU.calculate_preferred_global_size(entity_count);

        int hull_count        = core_memory.sector_container().next_hull();
        int edge_count        = core_memory.sector_container().next_edge();
        int point_count       = core_memory.sector_container().next_point();
        int hull_bone_count   = core_memory.sector_container().next_hull_bone();
        int entity_bone_count = core_memory.sector_container().next_entity_bone();

        int hull_size        = GPGPU.calculate_preferred_global_size(hull_count);
        int edge_size        = GPGPU.calculate_preferred_global_size(edge_count);
        int point_size       = GPGPU.calculate_preferred_global_size(point_count);
        int hull_bone_size   = GPGPU.calculate_preferred_global_size(hull_bone_count);
        int entity_bone_size = GPGPU.calculate_preferred_global_size(entity_bone_count);

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
            .call(arg_long(entity_size), GPGPU.preferred_work_size);

        k_egress_hulls
            .set_arg(EgressHulls_k.Args.max_hull, hull_count)
            .call(arg_long(hull_size), GPGPU.preferred_work_size);

        k_egress_edges
            .set_arg(EgressEdges_k.Args.max_edge, edge_count)
            .call(arg_long(edge_size), GPGPU.preferred_work_size);

        k_egress_points
            .set_arg(EgressPoints_k.Args.max_point, point_count)
            .call(arg_long(point_size), GPGPU.preferred_work_size);

        k_egress_hull_bones
            .set_arg(EgressHullBones_k.Args.max_hull_bone, hull_bone_count)
            .call(arg_long(hull_bone_size), GPGPU.preferred_work_size);

        k_egress_entity_bones
            .set_arg(EgressEntityBones_k.Args.max_entity_bone, entity_bone_count)
            .call(arg_long(entity_bone_size), GPGPU.preferred_work_size);
    }

    public void unload(UnorderedCoreBufferGroup.Raw raw_sectors, int[] counts)
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
