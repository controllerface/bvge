package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CompleteIntMultiBlockOut_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.scan_int_array_out;
    private static final GPU.Kernel kernel = GPU.Kernel.complete_int_multi_block_out;

    public enum Args
    {
        output,
        buffer,
        part,
        n;
    }

    public CompleteIntMultiBlockOut_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
