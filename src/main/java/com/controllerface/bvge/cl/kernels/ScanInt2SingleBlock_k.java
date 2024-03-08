package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class ScanInt2SingleBlock_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.scan_int2_array;
    private static final GPU.Kernel kernel = GPU.Kernel.scan_int2_single_block;

    public enum Args
    {
        data,
        buffer,
        n;
    }

    public ScanInt2SingleBlock_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
