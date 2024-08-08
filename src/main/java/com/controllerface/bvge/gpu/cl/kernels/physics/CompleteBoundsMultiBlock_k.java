package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class CompleteBoundsMultiBlock_k extends GPUKernel
{
    public enum Args
    {
        bounds_bank_data,
        sz,
        buffer,
        part,
        n;
    }

    public CompleteBoundsMultiBlock_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.complete_bounds_multi_block));
    }
}
