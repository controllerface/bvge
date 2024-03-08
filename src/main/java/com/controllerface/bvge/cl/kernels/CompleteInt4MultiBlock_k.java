package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CompleteInt4MultiBlock_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.scan_int4_array;
    private static final GPU.Kernel kernel = GPU.Kernel.complete_int4_multi_block;

    public enum Args
    {
        data,
        buffer,
        part,
        n;
    }

    public CompleteInt4MultiBlock_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
