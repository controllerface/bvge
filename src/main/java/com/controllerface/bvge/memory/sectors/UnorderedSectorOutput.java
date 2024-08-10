package com.controllerface.bvge.memory.sectors;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.GPUResource;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.buffers.TransientBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.egress.*;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.crud.GPUCrud;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.groups.UnorderedCoreBufferGroup;

import static com.controllerface.bvge.gpu.GPU.CL.arg_long;
import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.cl_int;

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

    public UnorderedSectorOutput(CL_CommandQueue cmd_queue,
                                 GPUCoreMemory core_memory,
                                 String name,
                                 long entity_init,
                                 long hull_init,
                                 long edge_init,
                                 long point_init)
    {
        this.cmd_queue        = cmd_queue;
        this.core_memory      = core_memory;
        this.egress_sizes_buf = GPU.CL.new_pinned_buffer(GPU.compute.context, (long)cl_int.size() * 6);
        this.sector_buffers   = new UnorderedCoreBufferGroup(name, this.cmd_queue, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        this.p_gpu_crud       = new GPUCrud().init();

        b_hull_shift        = new TransientBuffer(cmd_queue, cl_int.size(), hull_init);
        b_edge_shift        = new TransientBuffer(cmd_queue, cl_int.size(), edge_init);
        b_point_shift       = new TransientBuffer(cmd_queue, cl_int.size(), point_init);
        b_hull_bone_shift   = new TransientBuffer(cmd_queue, cl_int.size(), hull_init);
        b_entity_bone_shift = new TransientBuffer(cmd_queue, cl_int.size(), entity_init);

        k_egress_entities = new EgressEntities_k(cmd_queue, p_gpu_crud)
            .init(core_memory, sector_buffers, b_point_shift, b_edge_shift, b_hull_shift, b_hull_bone_shift, b_entity_bone_shift, egress_sizes_buf);

        k_egress_hulls = new EgressHulls_k(cmd_queue, p_gpu_crud)
            .init(core_memory, sector_buffers, b_hull_shift);

        k_egress_edges = new EgressEdges_k(cmd_queue, p_gpu_crud)
            .init(core_memory, sector_buffers, b_edge_shift);

        k_egress_points = new EgressPoints_k(cmd_queue, p_gpu_crud)
            .init(core_memory, sector_buffers, b_point_shift);

        k_egress_hull_bones = new EgressHullBones_k(cmd_queue, p_gpu_crud)
            .init(core_memory, sector_buffers, b_hull_bone_shift);

        k_egress_entity_bones = new EgressEntityBones_k(cmd_queue, p_gpu_crud)
            .init(core_memory, sector_buffers, b_entity_bone_shift);
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

        int entity_size = GPU.compute.calculate_preferred_global_size(entity_count);

        int hull_count        = core_memory.sector_container().next_hull();
        int edge_count        = core_memory.sector_container().next_edge();
        int point_count       = core_memory.sector_container().next_point();
        int hull_bone_count   = core_memory.sector_container().next_hull_bone();
        int entity_bone_count = core_memory.sector_container().next_entity_bone();

        int hull_size        = GPU.compute.calculate_preferred_global_size(hull_count);
        int edge_size        = GPU.compute.calculate_preferred_global_size(edge_count);
        int point_size       = GPU.compute.calculate_preferred_global_size(point_count);
        int hull_bone_size   = GPU.compute.calculate_preferred_global_size(hull_bone_count);
        int entity_bone_size = GPU.compute.calculate_preferred_global_size(entity_bone_count);

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
            .call(arg_long(entity_size), GPU.compute.preferred_work_size);

        k_egress_hulls
            .set_arg(EgressHulls_k.Args.max_hull, hull_count)
            .call(arg_long(hull_size), GPU.compute.preferred_work_size);

        k_egress_edges
            .set_arg(EgressEdges_k.Args.max_edge, edge_count)
            .call(arg_long(edge_size), GPU.compute.preferred_work_size);

        k_egress_points
            .set_arg(EgressPoints_k.Args.max_point, point_count)
            .call(arg_long(point_size), GPU.compute.preferred_work_size);

        k_egress_hull_bones
            .set_arg(EgressHullBones_k.Args.max_hull_bone, hull_bone_count)
            .call(arg_long(hull_bone_size), GPU.compute.preferred_work_size);

        k_egress_entity_bones
            .set_arg(EgressEntityBones_k.Args.max_entity_bone, entity_bone_count)
            .call(arg_long(entity_bone_size), GPU.compute.preferred_work_size);
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
