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

public class CompactPoints_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.compact_k_src(KernelType.compact_points, Args.class);

    public enum Args implements KernelArg
    {
        point_shift             (CL_DataTypes.cl_int.buffer_name()),
        points                  (POINT.data_type().buffer_name()),
        anti_gravity            (POINT_ANTI_GRAV.data_type().buffer_name()),
        anti_time               (POINT_ANTI_TIME.data_type().buffer_name()),
        point_vertex_references (POINT_VERTEX_REFERENCE.data_type().buffer_name()),
        point_hull_indices      (POINT_HULL_INDEX.data_type().buffer_name()),
        point_flags             (POINT_FLAG.data_type().buffer_name()),
        point_hit_counts        (POINT_HIT_COUNT.data_type().buffer_name()),
        bone_tables             (POINT_BONE_TABLE.data_type().buffer_name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CompactPoints_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.compact_points));
    }

    public GPUKernel init(CoreBufferGroup sector_buffers, ResizableBuffer b_point_shift)
    {
        return this.buf_arg(Args.point_shift, b_point_shift)
            .buf_arg(Args.points, sector_buffers.buffer(POINT))
            .buf_arg(Args.anti_gravity, sector_buffers.buffer(POINT_ANTI_GRAV))
            .buf_arg(Args.anti_time, sector_buffers.buffer(POINT_ANTI_TIME))
            .buf_arg(Args.point_vertex_references, sector_buffers.buffer(POINT_VERTEX_REFERENCE))
            .buf_arg(Args.point_hull_indices, sector_buffers.buffer(POINT_HULL_INDEX))
            .buf_arg(Args.point_flags, sector_buffers.buffer(POINT_FLAG))
            .buf_arg(Args.point_hit_counts, sector_buffers.buffer(POINT_HIT_COUNT))
            .buf_arg(Args.bone_tables, sector_buffers.buffer(POINT_BONE_TABLE));
    }
}
