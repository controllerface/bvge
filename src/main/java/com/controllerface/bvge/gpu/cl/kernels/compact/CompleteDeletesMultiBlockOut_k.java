package com.controllerface.bvge.gpu.cl.kernels.compact;

import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class CompleteDeletesMultiBlockOut_k extends GPUKernel
{
    public enum Args
    {
        entity_flags,
        hull_tables,
        bone_tables,
        point_tables,
        edge_tables,
        hull_bone_tables,
        output1,
        output2,
        sz,
        buffer1,
        buffer2,
        part1,
        part2,
        n,
    }

    public CompleteDeletesMultiBlockOut_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.complete_deletes_multi_block_out));
    }

    public GPUKernel init(CoreBufferGroup sector_buffers, CL_Buffer delete_sizes, ResizableBuffer b_delete_partial_1, ResizableBuffer b_delete_partial_2)
    {
        return this.buf_arg(Args.sz, delete_sizes)
            .buf_arg(Args.part1, b_delete_partial_1)
            .buf_arg(Args.part2, b_delete_partial_2)
            .buf_arg(Args.entity_flags, sector_buffers.buffer(ENTITY_FLAG))
            .buf_arg(Args.hull_tables, sector_buffers.buffer(ENTITY_HULL_TABLE))
            .buf_arg(Args.bone_tables, sector_buffers.buffer(ENTITY_BONE_TABLE))
            .buf_arg(Args.point_tables, sector_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(Args.edge_tables, sector_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(Args.hull_bone_tables, sector_buffers.buffer(HULL_BONE_TABLE));
    }
}
