package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.ENTITY_FLAG;

public class ClearSelectBlock_k extends GPUKernel
{
    public enum Args
    {
        entity_flags,
        target,
    }

    public ClearSelectBlock_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.clear_select_block));
    }

    public GPUKernel init(CoreBufferGroup core_buffers)
    {
        return this.buf_arg(Args.entity_flags, core_buffers.buffer(ENTITY_FLAG));
    }
}
