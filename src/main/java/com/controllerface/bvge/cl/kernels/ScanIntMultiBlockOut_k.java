package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class ScanIntMultiBlockOut_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.scan_int_array_out;
    private static final GPU.Kernel kernel = GPU.Kernel.scan_int_multi_block_out;

    public enum Args
    {
        input,
        output,
        buffer,
        part,
        n;
    }

    public ScanIntMultiBlockOut_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
