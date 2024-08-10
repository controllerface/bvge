package com.controllerface.bvge.gpu.cl.kernels.compact;

import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class ScanDeletesSingleBlockOut_k extends GPUKernel
{
    public enum Args
    {
        entity_flags,
        hull_tables,
        bone_tables,
        point_tables,
        edge_tables,
        hull_bone_tables,
        output,
        output2,
        sz,
        buffer,
        buffer2,
        n,
    }

    public ScanDeletesSingleBlockOut_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.scan_deletes_single_block_out));
    }

    public GPUKernel init(CoreBufferGroup sector_buffers, CL_Buffer delete_sizes)
    {
        return this.buf_arg(Args.sz, delete_sizes)
            .buf_arg(Args.entity_flags, sector_buffers.buffer(ENTITY_FLAG))
            .buf_arg(Args.hull_tables, sector_buffers.buffer(ENTITY_HULL_TABLE))
            .buf_arg(Args.bone_tables, sector_buffers.buffer(ENTITY_BONE_TABLE))
            .buf_arg(Args.point_tables, sector_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(Args.edge_tables, sector_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(Args.hull_bone_tables, sector_buffers.buffer(HULL_BONE_TABLE));
    }
}
