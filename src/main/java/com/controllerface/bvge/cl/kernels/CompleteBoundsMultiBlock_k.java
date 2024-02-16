package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CompleteBoundsMultiBlock_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.scan_key_bank;
    private static final GPU.Kernel kernel = GPU.Kernel.complete_bounds_multi_block;

    public enum Args
    {
        bounds_bank_data,
        sz,
        buffer,
        part,
        n;
    }

    public CompleteBoundsMultiBlock_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
