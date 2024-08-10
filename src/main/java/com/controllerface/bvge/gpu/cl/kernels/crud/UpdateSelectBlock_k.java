package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class UpdateSelectBlock_k extends GPUKernel
{
    public enum Args
    {
        entity_flags,
        hull_uv_offsets,
        entity_hull_tables,
        target,
        new_value,
    }

    public UpdateSelectBlock_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.update_select_block));
    }

    public GPUKernel init(CoreBufferGroup core_buffers)
    {
        return this.buf_arg(Args.entity_flags, core_buffers.buffer(ENTITY_FLAG))
            .buf_arg(Args.hull_uv_offsets, core_buffers.buffer(HULL_UV_OFFSET))
            .buf_arg(Args.entity_hull_tables, core_buffers.buffer(ENTITY_HULL_TABLE));
    }
}
