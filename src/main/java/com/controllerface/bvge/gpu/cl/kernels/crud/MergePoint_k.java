package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class MergePoint_k extends GPUKernel
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
        point_offset,
        bone_offset,
        hull_offset,
        max_point,
    }

    public MergePoint_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.merge_point));
    }

    public GPUKernel init(GPUCoreMemory core_memory, CoreBufferGroup core_buffers)
    {
        return this.buf_arg(Args.points_in, core_buffers.buffer(POINT))
            .buf_arg(Args.point_vertex_references_in, core_buffers.buffer(POINT_VERTEX_REFERENCE))
            .buf_arg(Args.point_hull_indices_in, core_buffers.buffer(POINT_HULL_INDEX))
            .buf_arg(Args.point_hit_counts_in, core_buffers.buffer(POINT_HIT_COUNT))
            .buf_arg(Args.point_flags_in, core_buffers.buffer(POINT_FLAG))
            .buf_arg(Args.point_bone_tables_in, core_buffers.buffer(POINT_BONE_TABLE))
            .buf_arg(Args.points_out, core_memory.get_buffer(POINT))
            .buf_arg(Args.point_vertex_references_out, core_memory.get_buffer(POINT_VERTEX_REFERENCE))
            .buf_arg(Args.point_hull_indices_out, core_memory.get_buffer(POINT_HULL_INDEX))
            .buf_arg(Args.point_hit_counts_out, core_memory.get_buffer(POINT_HIT_COUNT))
            .buf_arg(Args.point_flags_out, core_memory.get_buffer(POINT_FLAG))
            .buf_arg(Args.point_bone_tables_out, core_memory.get_buffer(POINT_BONE_TABLE));
    }
}
