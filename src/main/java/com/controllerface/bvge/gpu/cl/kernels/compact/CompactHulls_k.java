package com.controllerface.bvge.gpu.cl.kernels.compact;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class CompactHulls_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.compact_k_src(KernelType.compact_hulls, Args.class);

    public enum Args implements KernelArg
    {
        hull_shift          (CL_DataTypes.cl_int.buffer_name()),
        hulls               (HULL.data_type().buffer_name()),
        hull_scales         (HULL_SCALE.data_type().buffer_name()),
        hull_mesh_ids       (HULL_MESH_ID.data_type().buffer_name()),
        hull_uv_offsets     (HULL_UV_OFFSET.data_type().buffer_name()),
        hull_rotations      (HULL_ROTATION.data_type().buffer_name()),
        hull_frictions      (HULL_FRICTION.data_type().buffer_name()),
        hull_restitutions   (HULL_RESTITUTION.data_type().buffer_name()),
        hull_integrity      (HULL_INTEGRITY.data_type().buffer_name()),
        hull_bone_tables    (HULL_BONE_TABLE.data_type().buffer_name()),
        hull_entity_ids     (HULL_ENTITY_ID.data_type().buffer_name()),
        hull_flags          (HULL_FLAG.data_type().buffer_name()),
        hull_point_tables   (HULL_POINT_TABLE.data_type().buffer_name()),
        hull_edge_tables    (HULL_EDGE_TABLE.data_type().buffer_name()),
        hull_aabb           (HULL_AABB.data_type().buffer_name()),
        hull_aabb_index     (HULL_AABB_INDEX.data_type().buffer_name()),
        hull_aabb_key_table (HULL_AABB_KEY_TABLE.data_type().buffer_name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CompactHulls_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.compact_hulls));
    }

    public GPUKernel init(CoreBufferGroup sector_buffers, ResizableBuffer b_hull_shift)
    {
        return this.buf_arg(Args.hull_shift, b_hull_shift)
            .buf_arg(Args.hulls, sector_buffers.buffer(HULL))
            .buf_arg(Args.hull_scales, sector_buffers.buffer(HULL_SCALE))
            .buf_arg(Args.hull_mesh_ids, sector_buffers.buffer(HULL_MESH_ID))
            .buf_arg(Args.hull_uv_offsets, sector_buffers.buffer(HULL_UV_OFFSET))
            .buf_arg(Args.hull_rotations, sector_buffers.buffer(HULL_ROTATION))
            .buf_arg(Args.hull_frictions, sector_buffers.buffer(HULL_FRICTION))
            .buf_arg(Args.hull_restitutions, sector_buffers.buffer(HULL_RESTITUTION))
            .buf_arg(Args.hull_integrity, sector_buffers.buffer(HULL_INTEGRITY))
            .buf_arg(Args.hull_bone_tables, sector_buffers.buffer(HULL_BONE_TABLE))
            .buf_arg(Args.hull_entity_ids, sector_buffers.buffer(HULL_ENTITY_ID))
            .buf_arg(Args.hull_flags, sector_buffers.buffer(HULL_FLAG))
            .buf_arg(Args.hull_point_tables, sector_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(Args.hull_edge_tables, sector_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(Args.hull_aabb, sector_buffers.buffer(HULL_AABB))
            .buf_arg(Args.hull_aabb_index, sector_buffers.buffer(HULL_AABB_INDEX))
            .buf_arg(Args.hull_aabb_key_table, sector_buffers.buffer(HULL_AABB_KEY_TABLE));
    }
}
