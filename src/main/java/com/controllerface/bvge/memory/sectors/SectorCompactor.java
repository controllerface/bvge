package com.controllerface.bvge.memory.sectors;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.GPUResource;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.buffers.TransientBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.compact.*;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.scan.GPUScanVectorInt2;
import com.controllerface.bvge.gpu.cl.programs.scan.GPUScanVectorInt4;
import com.controllerface.bvge.gpu.cl.programs.scan.ScanDeletes;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.gpu.GPU.CL.arg_long;
import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.*;

public class SectorCompactor implements GPUResource
{
    private static final int DELETE_COUNTERS = 6;
    private static final int DELETE_COUNTERS_SIZE = cl_int.size() * DELETE_COUNTERS;

    private final GPUProgram p_scan_deletes;

    private final GPUKernel k_compact_armature_bones;
    private final GPUKernel k_compact_edges;
    private final GPUKernel k_compact_entities;
    private final GPUKernel k_compact_hull_bones;
    private final GPUKernel k_compact_hulls;
    private final GPUKernel k_compact_points;
    private final GPUKernel k_complete_deletes_multi_block_out;
    private final GPUKernel k_scan_deletes_multi_block_out;
    private final GPUKernel k_scan_deletes_single_block_out;

    /**
     * During the entity compaction process, these buffers are written to, and store the number of
     * positions that the corresponding values must shift left within their own buffers when the
     * buffer compaction occurs. Each index is aligned with the corresponding data type
     * that will be shifted. I.e. every bone in the bone buffer has a corresponding entry in the
     * bone shift buffer. Points, edges, and hulls work the same way.
     */

    private final ResizableBuffer b_entity_bone_shift;
    private final ResizableBuffer b_hull_bone_shift;
    private final ResizableBuffer b_edge_shift;
    private final ResizableBuffer b_hull_shift;
    private final ResizableBuffer b_point_shift;

    /**
     * During the deletion process, these buffers are used during the parallel scan of the relevant data
     * buffers. The partial buffers are utilized when the parallel scan occurs over multiple scan blocks,
     * and allows the output of each block to then itself be scanned, until all values have been summed.
     */

    private final ResizableBuffer b_delete_1;
    private final ResizableBuffer b_delete_2;
    private final ResizableBuffer b_delete_partial_1;
    private final ResizableBuffer b_delete_partial_2;

    private final CL_Buffer delete_sizes_buf;
    private final SectorController controller;

    private final GPUScanVectorInt2 gpu_int2_scan;
    private final GPUScanVectorInt4 gpu_int4_scan;

    private final CL_CommandQueue cmd_queue;

    public SectorCompactor(CL_CommandQueue cmd_queue,
                           SectorController controller,
                           CoreBufferGroup sector_buffers,
                           long entity_init,
                           long hull_init,
                           long edge_init,
                           long point_init,
                           long delete_init)
    {
        this.p_scan_deletes = new ScanDeletes().init();
        this.cmd_queue = cmd_queue;
        this.gpu_int2_scan = new GPUScanVectorInt2(cmd_queue);
        this.gpu_int4_scan = new GPUScanVectorInt4(cmd_queue);
        this.controller = controller;
        delete_sizes_buf = GPU.CL.new_pinned_buffer(GPU.compute.context, DELETE_COUNTERS_SIZE);

        b_hull_shift        = new TransientBuffer(cmd_queue, cl_int.size(),  hull_init);
        b_edge_shift        = new TransientBuffer(cmd_queue, cl_int.size(),  edge_init);
        b_point_shift       = new TransientBuffer(cmd_queue, cl_int.size(),  point_init);
        b_hull_bone_shift   = new TransientBuffer(cmd_queue, cl_int.size(),  hull_init);
        b_entity_bone_shift = new TransientBuffer(cmd_queue, cl_int.size(),  entity_init);
        b_delete_1          = new TransientBuffer(cmd_queue, cl_int2.size(), delete_init);
        b_delete_2          = new TransientBuffer(cmd_queue, cl_int4.size(), delete_init);
        b_delete_partial_1  = new TransientBuffer(cmd_queue, cl_int2.size(), delete_init);
        b_delete_partial_2  = new TransientBuffer(cmd_queue, cl_int4.size(), delete_init);

        k_scan_deletes_single_block_out = new ScanDeletesSingleBlockOut_k(cmd_queue, p_scan_deletes)
            .init(sector_buffers, delete_sizes_buf);

        k_scan_deletes_multi_block_out = new ScanDeletesMultiBlockOut_k(cmd_queue, p_scan_deletes)
            .init(sector_buffers, b_delete_partial_1, b_delete_partial_2);

        k_complete_deletes_multi_block_out = new CompleteDeletesMultiBlockOut_k(cmd_queue, p_scan_deletes)
            .init(sector_buffers, delete_sizes_buf, b_delete_partial_1, b_delete_partial_2);

        k_compact_entities = new CompactEntities_k(cmd_queue, p_scan_deletes)
            .init(sector_buffers, b_entity_bone_shift, b_hull_bone_shift, b_edge_shift, b_hull_shift, b_point_shift);

        k_compact_hulls = new CompactHulls_k(cmd_queue, p_scan_deletes)
            .init(sector_buffers, b_hull_shift);

        k_compact_edges = new CompactEdges_k(cmd_queue, p_scan_deletes)
            .init(sector_buffers, b_edge_shift);

        k_compact_points = new CompactPoints_k(cmd_queue, p_scan_deletes)
            .init(sector_buffers, b_point_shift);

        k_compact_hull_bones = new CompactHullBones_k(cmd_queue, p_scan_deletes)
            .init(sector_buffers, b_hull_bone_shift);

        k_compact_armature_bones = new CompactEntityBones_k(cmd_queue, p_scan_deletes)
            .init(sector_buffers, b_entity_bone_shift);
    }

    private void linearize_kernel(GPUKernel kernel, int object_count)
    {
        int offset = 0;
        for (long remaining = object_count; remaining > 0; remaining -= GPU.compute.max_work_group_size)
        {
            int count = (int) Math.min(GPU.compute.max_work_group_size, remaining);
            var sz = count == GPU.compute.max_work_group_size
                ? GPU.compute.local_work_default
                : arg_long(count);
            kernel.call(sz, sz, arg_long(offset));
            offset += count;
        }
    }

    private int[] scan_single_block_deletes_out(long o1_data_ptr, long o2_data_ptr, int n)
    {
        long local_buffer_size = cl_int2.size() * GPU.compute.max_scan_block_size;
        long local_buffer_size2 = cl_int4.size() * GPU.compute.max_scan_block_size;

        GPU.CL.zero_buffer(cmd_queue, delete_sizes_buf, DELETE_COUNTERS_SIZE);

        k_scan_deletes_single_block_out
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.output, o1_data_ptr)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(ScanDeletesSingleBlockOut_k.Args.buffer, local_buffer_size)
            .loc_arg(ScanDeletesSingleBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(ScanDeletesSingleBlockOut_k.Args.n, n)
            .call(GPU.compute.local_work_default, GPU.compute.local_work_default);

        return GPU.CL.read_pinned_int_buffer(cmd_queue, delete_sizes_buf, cl_int.size(), DELETE_COUNTERS);
    }

    private int[] scan_multi_block_deletes_out(long o1_data_ptr, long o2_data_ptr, int n, int k)
    {
        long local_buffer_size = cl_int2.size() * GPU.compute.max_scan_block_size;
        long local_buffer_size2 = cl_int4.size() * GPU.compute.max_scan_block_size;

        long gx = k * GPU.compute.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;

        b_delete_partial_1.ensure_capacity(part_size);
        b_delete_partial_2.ensure_capacity(part_size);

        k_scan_deletes_multi_block_out
            .ptr_arg(ScanDeletesMultiBlockOut_k.Args.output1, o1_data_ptr)
            .ptr_arg(ScanDeletesMultiBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(ScanDeletesMultiBlockOut_k.Args.buffer1, local_buffer_size)
            .loc_arg(ScanDeletesMultiBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(ScanDeletesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPU.compute.local_work_default);

        // note the partial buffers are scanned and updated in-place
        gpu_int2_scan.scan_int2(b_delete_partial_1.pointer(), part_size);
        gpu_int4_scan.scan_int4(b_delete_partial_2.pointer(), part_size);

        GPU.CL.zero_buffer(cmd_queue, delete_sizes_buf, DELETE_COUNTERS_SIZE);

        k_complete_deletes_multi_block_out
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.output1, o1_data_ptr)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(CompleteDeletesMultiBlockOut_k.Args.buffer1, local_buffer_size)
            .loc_arg(CompleteDeletesMultiBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(CompleteDeletesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPU.compute.local_work_default);

        return GPU.CL.read_pinned_int_buffer(cmd_queue, delete_sizes_buf, cl_int.size(), DELETE_COUNTERS);
    }

    public int[] scan_deletes(long o1_data_ptr, long o2_data_ptr, int n)
    {
        int k = GPU.compute.work_group_count(n);
        if (k == 1)
        {
            return scan_single_block_deletes_out(o1_data_ptr, o2_data_ptr, n);
        }
        else
        {
            return scan_multi_block_deletes_out(o1_data_ptr, o2_data_ptr, n, k);
        }
    }

    public void delete_and_compact()
    {
        b_delete_1.ensure_capacity(controller.next_entity());
        b_delete_2.ensure_capacity(controller.next_entity());

        int[] shift_counts = scan_deletes(b_delete_1.pointer(), b_delete_2.pointer(), controller.next_entity());

        if (shift_counts[4] == 0)
        {
            return;
        }

        b_hull_shift.ensure_capacity(controller.next_hull());
        b_edge_shift.ensure_capacity(controller.next_edge());
        b_point_shift.ensure_capacity(controller.next_point());
        b_hull_bone_shift.ensure_capacity(controller.next_hull_bone());
        b_entity_bone_shift.ensure_capacity(controller.next_entity_bone());

        b_hull_shift.clear();
        b_edge_shift.clear();
        b_point_shift.clear();
        b_hull_bone_shift.clear();
        b_entity_bone_shift.clear();

        k_compact_entities
            .ptr_arg(CompactEntities_k.Args.buffer_in_1, b_delete_1.pointer())
            .ptr_arg(CompactEntities_k.Args.buffer_in_2, b_delete_2.pointer());

        linearize_kernel(k_compact_entities, controller.next_entity());
        linearize_kernel(k_compact_hull_bones, controller.next_hull_bone());
        linearize_kernel(k_compact_points, controller.next_point());
        linearize_kernel(k_compact_edges, controller.next_edge());
        linearize_kernel(k_compact_hulls, controller.next_hull());
        linearize_kernel(k_compact_armature_bones, controller.next_entity_bone());

        compact_buffers(shift_counts);
    }

    private void compact_buffers(int[] shift_counts)
    {
        controller.compact(shift_counts);
    }

    public void release()
    {
        delete_sizes_buf.release();
        p_scan_deletes.release();
        gpu_int2_scan.release();
        gpu_int4_scan.release();

        b_entity_bone_shift.release();
        b_hull_bone_shift.release();
        b_edge_shift.release();
        b_hull_shift.release();
        b_point_shift.release();
        b_delete_1.release();
        b_delete_2.release();
        b_delete_partial_1.release();
        b_delete_partial_2.release();

        long usage = b_entity_bone_shift.debug_data()
            + b_hull_bone_shift.debug_data()
            + b_edge_shift.debug_data()
            + b_hull_shift.debug_data()
            + b_point_shift.debug_data()
            + b_delete_1.debug_data()
            + b_delete_2.debug_data()
            + b_delete_partial_1.debug_data()
            + b_delete_partial_2.debug_data();

        System.out.println("BufferGroup [Sector Compactor] Memory Usage: MB " + ((float) usage / 1024f / 1024f));
    }
}
