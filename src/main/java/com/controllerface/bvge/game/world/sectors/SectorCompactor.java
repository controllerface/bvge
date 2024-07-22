package com.controllerface.bvge.game.world.sectors;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.cl.GPUScanVectorInt2;
import com.controllerface.bvge.cl.GPUScanVectorInt4;
import com.controllerface.bvge.cl.buffers.Destroyable;
import com.controllerface.bvge.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.cl.buffers.CoreBufferGroup;
import com.controllerface.bvge.cl.buffers.TransientBuffer;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.kernels.compact.*;
import com.controllerface.bvge.cl.programs.GPUProgram;
import com.controllerface.bvge.cl.programs.ScanDeletes;

import static com.controllerface.bvge.cl.CLData.*;
import static com.controllerface.bvge.cl.CLUtils.arg_long;
import static com.controllerface.bvge.cl.buffers.CoreBufferType.*;

public class SectorCompactor implements Destroyable
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

    private final long ptr_delete_sizes;
    private final SectorController controller;

    private final GPUScanVectorInt2 gpu_int2_scan;
    private final GPUScanVectorInt4 gpu_int4_scan;

    private final long ptr_queue;

    public SectorCompactor(long ptr_queue,
                           SectorController controller,
                           CoreBufferGroup sector_buffers,
                           long entity_init,
                           long hull_init,
                           long edge_init,
                           long point_init,
                           long delete_init)
    {
        this.p_scan_deletes = new ScanDeletes().init();
        this.ptr_queue = ptr_queue;
        this.gpu_int2_scan = new GPUScanVectorInt2(ptr_queue);
        this.gpu_int4_scan = new GPUScanVectorInt4(ptr_queue);
        this.controller = controller;
        ptr_delete_sizes    = GPGPU.cl_new_pinned_buffer(DELETE_COUNTERS_SIZE);

        b_hull_shift        = new TransientBuffer(ptr_queue, cl_int.size(),  hull_init);
        b_edge_shift        = new TransientBuffer(ptr_queue, cl_int.size(),  edge_init);
        b_point_shift       = new TransientBuffer(ptr_queue, cl_int.size(),  point_init);
        b_hull_bone_shift   = new TransientBuffer(ptr_queue, cl_int.size(),  hull_init);
        b_entity_bone_shift = new TransientBuffer(ptr_queue, cl_int.size(),  entity_init);
        b_delete_1          = new TransientBuffer(ptr_queue, cl_int2.size(), delete_init);
        b_delete_2          = new TransientBuffer(ptr_queue, cl_int4.size(), delete_init);
        b_delete_partial_1  = new TransientBuffer(ptr_queue, cl_int2.size(), delete_init);
        b_delete_partial_2  = new TransientBuffer(ptr_queue, cl_int4.size(), delete_init);

        long k_ptr_scan_deletes_single_block_out = p_scan_deletes.kernel_ptr(Kernel.scan_deletes_single_block_out);
        k_scan_deletes_single_block_out = new ScanDeletesSingleBlockOut_k(ptr_queue, k_ptr_scan_deletes_single_block_out)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.sz, ptr_delete_sizes)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.entity_flags,     sector_buffers.buffer(ENTITY_FLAG))
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.hull_tables,      sector_buffers.buffer(ENTITY_HULL_TABLE))
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.bone_tables,      sector_buffers.buffer(ENTITY_BONE_TABLE))
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.point_tables,     sector_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.edge_tables,      sector_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.hull_bone_tables, sector_buffers.buffer(HULL_BONE_TABLE));

        long k_ptr_scan_deletes_multi_block_out = p_scan_deletes.kernel_ptr(Kernel.scan_deletes_multi_block_out);
        k_scan_deletes_multi_block_out = new ScanDeletesMultiBlockOut_k(ptr_queue, k_ptr_scan_deletes_multi_block_out)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.part1,             b_delete_partial_1)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.part2,             b_delete_partial_2)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.entity_flags,      sector_buffers.buffer(ENTITY_FLAG))
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.hull_tables,       sector_buffers.buffer(ENTITY_HULL_TABLE))
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.bone_tables,       sector_buffers.buffer(ENTITY_BONE_TABLE))
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.point_tables,      sector_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.edge_tables,       sector_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.hull_bone_tables,  sector_buffers.buffer(HULL_BONE_TABLE));

        long k_ptr_complete_deletes_multi_block_out = p_scan_deletes.kernel_ptr(Kernel.complete_deletes_multi_block_out);
        k_complete_deletes_multi_block_out = new CompleteDeletesMultiBlockOut_k(ptr_queue, k_ptr_complete_deletes_multi_block_out)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.sz, ptr_delete_sizes)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.part1,             b_delete_partial_1)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.part2,             b_delete_partial_2)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.entity_flags,      sector_buffers.buffer(ENTITY_FLAG))
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.hull_tables,       sector_buffers.buffer(ENTITY_HULL_TABLE))
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.bone_tables,       sector_buffers.buffer(ENTITY_BONE_TABLE))
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.point_tables,      sector_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.edge_tables,       sector_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.hull_bone_tables,  sector_buffers.buffer(HULL_BONE_TABLE));

        long k_ptr_compact_entities = p_scan_deletes.kernel_ptr(Kernel.compact_entities);
        k_compact_entities = new CompactEntities_k(ptr_queue, k_ptr_compact_entities)
            .buf_arg(CompactEntities_k.Args.entities,                   sector_buffers.buffer(ENTITY))
            .buf_arg(CompactEntities_k.Args.entity_masses,              sector_buffers.buffer(ENTITY_MASS))
            .buf_arg(CompactEntities_k.Args.entity_root_hulls,          sector_buffers.buffer(ENTITY_ROOT_HULL))
            .buf_arg(CompactEntities_k.Args.entity_model_indices,       sector_buffers.buffer(ENTITY_MODEL_ID))
            .buf_arg(CompactEntities_k.Args.entity_model_transforms,    sector_buffers.buffer(ENTITY_TRANSFORM_ID))
            .buf_arg(CompactEntities_k.Args.entity_types,               sector_buffers.buffer(ENTITY_TYPE))
            .buf_arg(CompactEntities_k.Args.entity_flags,               sector_buffers.buffer(ENTITY_FLAG))
            .buf_arg(CompactEntities_k.Args.entity_animation_layers,    sector_buffers.buffer(ENTITY_ANIM_LAYER))
            .buf_arg(CompactEntities_k.Args.entity_previous_layers,     sector_buffers.buffer(ENTITY_PREV_LAYER))
            .buf_arg(CompactEntities_k.Args.entity_animation_time,      sector_buffers.buffer(ENTITY_ANIM_TIME))
            .buf_arg(CompactEntities_k.Args.entity_previous_time,       sector_buffers.buffer(ENTITY_PREV_TIME))
            .buf_arg(CompactEntities_k.Args.entity_animation_blend,     sector_buffers.buffer(ENTITY_ANIM_BLEND))
            .buf_arg(CompactEntities_k.Args.entity_motion_states,       sector_buffers.buffer(ENTITY_MOTION_STATE))
            .buf_arg(CompactEntities_k.Args.entity_entity_hull_tables,  sector_buffers.buffer(ENTITY_HULL_TABLE))
            .buf_arg(CompactEntities_k.Args.entity_bone_tables,         sector_buffers.buffer(ENTITY_BONE_TABLE))
            .buf_arg(CompactEntities_k.Args.hull_bone_tables,           sector_buffers.buffer(HULL_BONE_TABLE))
            .buf_arg(CompactEntities_k.Args.hull_entity_ids,            sector_buffers.buffer(HULL_ENTITY_ID))
            .buf_arg(CompactEntities_k.Args.hull_point_tables,          sector_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(CompactEntities_k.Args.hull_edge_tables,           sector_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(CompactEntities_k.Args.points,                     sector_buffers.buffer(POINT))
            .buf_arg(CompactEntities_k.Args.point_hull_indices,         sector_buffers.buffer(POINT_HULL_INDEX))
            .buf_arg(CompactEntities_k.Args.point_bone_tables,          sector_buffers.buffer(POINT_BONE_TABLE))
            .buf_arg(CompactEntities_k.Args.entity_bone_parent_ids,     sector_buffers.buffer(ENTITY_BONE_PARENT_ID))
            .buf_arg(CompactEntities_k.Args.hull_bind_pose_indices,     sector_buffers.buffer(HULL_BONE_BIND_POSE))
            .buf_arg(CompactEntities_k.Args.edges,                      sector_buffers.buffer(EDGE))
            .buf_arg(CompactEntities_k.Args.hull_bone_shift,            b_hull_bone_shift)
            .buf_arg(CompactEntities_k.Args.point_shift,                b_point_shift)
            .buf_arg(CompactEntities_k.Args.edge_shift,                 b_edge_shift)
            .buf_arg(CompactEntities_k.Args.hull_shift,                 b_hull_shift)
            .buf_arg(CompactEntities_k.Args.entity_bone_shift,          b_entity_bone_shift);

        long k_ptr_compact_hulls = p_scan_deletes.kernel_ptr(Kernel.compact_hulls);
        k_compact_hulls = new CompactHulls_k(ptr_queue, k_ptr_compact_hulls)
            .buf_arg(CompactHulls_k.Args.hull_shift,            b_hull_shift)
            .buf_arg(CompactHulls_k.Args.hulls,                 sector_buffers.buffer(HULL))
            .buf_arg(CompactHulls_k.Args.hull_scales,           sector_buffers.buffer(HULL_SCALE))
            .buf_arg(CompactHulls_k.Args.hull_mesh_ids,         sector_buffers.buffer(HULL_MESH_ID))
            .buf_arg(CompactHulls_k.Args.hull_uv_offsets,       sector_buffers.buffer(HULL_UV_OFFSET))
            .buf_arg(CompactHulls_k.Args.hull_rotations,        sector_buffers.buffer(HULL_ROTATION))
            .buf_arg(CompactHulls_k.Args.hull_frictions,        sector_buffers.buffer(HULL_FRICTION))
            .buf_arg(CompactHulls_k.Args.hull_restitutions,     sector_buffers.buffer(HULL_RESTITUTION))
            .buf_arg(CompactHulls_k.Args.hull_integrity,        sector_buffers.buffer(HULL_INTEGRITY))
            .buf_arg(CompactHulls_k.Args.hull_bone_tables,      sector_buffers.buffer(HULL_BONE_TABLE))
            .buf_arg(CompactHulls_k.Args.hull_entity_ids,       sector_buffers.buffer(HULL_ENTITY_ID))
            .buf_arg(CompactHulls_k.Args.hull_flags,            sector_buffers.buffer(HULL_FLAG))
            .buf_arg(CompactHulls_k.Args.hull_point_tables,     sector_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(CompactHulls_k.Args.hull_edge_tables,      sector_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(CompactHulls_k.Args.hull_aabb,             sector_buffers.buffer(HULL_AABB))
            .buf_arg(CompactHulls_k.Args.hull_aabb_index,       sector_buffers.buffer(HULL_AABB_INDEX))
            .buf_arg(CompactHulls_k.Args.hull_aabb_key_table,   sector_buffers.buffer(HULL_AABB_KEY_TABLE));

        long k_ptr_compact_edges = p_scan_deletes.kernel_ptr(Kernel.compact_edges);
        k_compact_edges = new CompactEdges_k(ptr_queue, k_ptr_compact_edges)
            .buf_arg(CompactEdges_k.Args.edge_shift,            b_edge_shift)
            .buf_arg(CompactEdges_k.Args.edges,                 sector_buffers.buffer(EDGE))
            .buf_arg(CompactEdges_k.Args.edge_lengths,          sector_buffers.buffer(EDGE_LENGTH))
            .buf_arg(CompactEdges_k.Args.edge_flags,            sector_buffers.buffer(EDGE_FLAG))
            .buf_arg(CompactEdges_k.Args.edge_pins,             sector_buffers.buffer(EDGE_PIN));

        long k_ptr_compact_points = p_scan_deletes.kernel_ptr(Kernel.compact_points);
        k_compact_points = new CompactPoints_k(ptr_queue, k_ptr_compact_points)
            .buf_arg(CompactPoints_k.Args.point_shift,              b_point_shift)
            .buf_arg(CompactPoints_k.Args.points,                   sector_buffers.buffer(POINT))
            .buf_arg(CompactPoints_k.Args.anti_gravity,             sector_buffers.buffer(POINT_ANTI_GRAV))
            .buf_arg(CompactPoints_k.Args.anti_time,                sector_buffers.buffer(POINT_ANTI_TIME))
            .buf_arg(CompactPoints_k.Args.point_vertex_references,  sector_buffers.buffer(POINT_VERTEX_REFERENCE))
            .buf_arg(CompactPoints_k.Args.point_hull_indices,       sector_buffers.buffer(POINT_HULL_INDEX))
            .buf_arg(CompactPoints_k.Args.point_flags,              sector_buffers.buffer(POINT_FLAG))
            .buf_arg(CompactPoints_k.Args.point_hit_counts,         sector_buffers.buffer(POINT_HIT_COUNT))
            .buf_arg(CompactPoints_k.Args.bone_tables,              sector_buffers.buffer(POINT_BONE_TABLE));

        long k_ptr_compact_hull_bones = p_scan_deletes.kernel_ptr(Kernel.compact_hull_bones);
        k_compact_hull_bones = new CompactHullBones_k(ptr_queue, k_ptr_compact_hull_bones)
            .buf_arg(CompactHullBones_k.Args.hull_bone_shift,               b_hull_bone_shift)
            .buf_arg(CompactHullBones_k.Args.hull_bones,                    sector_buffers.buffer(HULL_BONE))
            .buf_arg(CompactHullBones_k.Args.hull_bind_pose_indices,        sector_buffers.buffer(HULL_BONE_BIND_POSE))
            .buf_arg(CompactHullBones_k.Args.hull_inv_bind_pose_indices,    sector_buffers.buffer(HULL_BONE_INV_BIND_POSE));

        long k_ptr_compact_armature_bones = p_scan_deletes.kernel_ptr(Kernel.compact_entity_bones);
        k_compact_armature_bones = new CompactEntityBones_k(ptr_queue, k_ptr_compact_armature_bones)
            .buf_arg(CompactEntityBones_k.Args.entity_bone_shift,           b_entity_bone_shift)
            .buf_arg(CompactEntityBones_k.Args.entity_bones,                sector_buffers.buffer(ENTITY_BONE))
            .buf_arg(CompactEntityBones_k.Args.entity_bone_reference_ids,   sector_buffers.buffer(ENTITY_BONE_REFERENCE_ID))
            .buf_arg(CompactEntityBones_k.Args.entity_bone_parent_ids,      sector_buffers.buffer(ENTITY_BONE_PARENT_ID));
    }

    private void linearize_kernel(GPUKernel kernel, int object_count)
    {
        int offset = 0;
        for (long remaining = object_count; remaining > 0; remaining -= GPGPU.max_work_group_size)
        {
            int count = (int) Math.min(GPGPU.max_work_group_size, remaining);
            var sz = count == GPGPU.max_work_group_size
                ? GPGPU.local_work_default
                : arg_long(count);
            kernel.call(sz, sz, arg_long(offset));
            offset += count;
        }
    }

    private int[] scan_single_block_deletes_out(long o1_data_ptr, long o2_data_ptr, int n)
    {
        long local_buffer_size = cl_int2.size() * GPGPU.max_scan_block_size;
        long local_buffer_size2 = cl_int4.size() * GPGPU.max_scan_block_size;

        GPGPU.cl_zero_buffer(ptr_queue, ptr_delete_sizes, DELETE_COUNTERS_SIZE);

        k_scan_deletes_single_block_out
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.output, o1_data_ptr)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(ScanDeletesSingleBlockOut_k.Args.buffer, local_buffer_size)
            .loc_arg(ScanDeletesSingleBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(ScanDeletesSingleBlockOut_k.Args.n, n)
            .call(GPGPU.local_work_default, GPGPU.local_work_default);

        return GPGPU.cl_read_pinned_int_buffer(ptr_queue, ptr_delete_sizes, cl_int.size(), DELETE_COUNTERS);
    }

    private int[] scan_multi_block_deletes_out(long o1_data_ptr, long o2_data_ptr, int n, int k)
    {
        long local_buffer_size = cl_int2.size() * GPGPU.max_scan_block_size;
        long local_buffer_size2 = cl_int4.size() * GPGPU.max_scan_block_size;

        long gx = k * GPGPU.max_scan_block_size;
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
            .call(global_work_size, GPGPU.local_work_default);

        // note the partial buffers are scanned and updated in-place
        gpu_int2_scan.scan_int2(b_delete_partial_1.pointer(), part_size);
        gpu_int4_scan.scan_int4(b_delete_partial_2.pointer(), part_size);

        GPGPU.cl_zero_buffer(ptr_queue, ptr_delete_sizes, DELETE_COUNTERS_SIZE);

        k_complete_deletes_multi_block_out
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.output1, o1_data_ptr)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(CompleteDeletesMultiBlockOut_k.Args.buffer1, local_buffer_size)
            .loc_arg(CompleteDeletesMultiBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(CompleteDeletesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        return GPGPU.cl_read_pinned_int_buffer(ptr_queue, ptr_delete_sizes, cl_int.size(), DELETE_COUNTERS);
    }

    public int[] scan_deletes(long o1_data_ptr, long o2_data_ptr, int n)
    {
        int k = GPGPU.work_group_count(n);
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

    public void destroy()
    {
        GPGPU.cl_release_buffer(ptr_delete_sizes);
        p_scan_deletes.destroy();
        gpu_int2_scan.destroy();
        gpu_int4_scan.destroy();

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
