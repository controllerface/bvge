package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class ScanIntSingleBlockOut_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.scan_int_array_out;
    private static final GPU.Kernel kernel = GPU.Kernel.scan_int_single_block_out;

    public enum Args
    {
        input,
        output,
        buffer,
        n;
    }

    public ScanIntSingleBlockOut_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
