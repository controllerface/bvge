package com.controllerface.bvge.gpu.cl.kernels.egress;

import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.groups.UnorderedCoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class EgressPoints_k extends GPUKernel
{
    public enum Args
    {
        points_in,
        point_vertex_references_in,
        point_hull_indices_in,
        point_hit_counts_in,
        point_flags_in,
        point_bone_tables_in,
        points_out,
        point_vertex_references_out,
        point_hull_indices_out,
        point_hit_counts_out,
        point_flags_out,
        point_bone_tables_out,
        new_points,
        max_point,
    }

    public EgressPoints_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.egress_points));
    }

    public GPUKernel init(GPUCoreMemory core_memory,
                          UnorderedCoreBufferGroup sector_buffers,
                          ResizableBuffer b_point_shift)
    {
        return this.buf_arg(Args.points_in, core_memory.get_buffer(POINT))
            .buf_arg(Args.point_vertex_references_in, core_memory.get_buffer(POINT_VERTEX_REFERENCE))
            .buf_arg(Args.point_hull_indices_in, core_memory.get_buffer(POINT_HULL_INDEX))
            .buf_arg(Args.point_hit_counts_in, core_memory.get_buffer(POINT_HIT_COUNT))
            .buf_arg(Args.point_flags_in, core_memory.get_buffer(POINT_FLAG))
            .buf_arg(Args.point_bone_tables_in, core_memory.get_buffer(POINT_BONE_TABLE))
            .buf_arg(Args.points_out, sector_buffers.buffer(POINT))
            .buf_arg(Args.point_vertex_references_out, sector_buffers.buffer(POINT_VERTEX_REFERENCE))
            .buf_arg(Args.point_hull_indices_out, sector_buffers.buffer(POINT_HULL_INDEX))
            .buf_arg(Args.point_hit_counts_out, sector_buffers.buffer(POINT_HIT_COUNT))
            .buf_arg(Args.point_flags_out, sector_buffers.buffer(POINT_FLAG))
            .buf_arg(Args.point_bone_tables_out, sector_buffers.buffer(POINT_BONE_TABLE))
            .buf_arg(Args.new_points, b_point_shift);
    }
}
