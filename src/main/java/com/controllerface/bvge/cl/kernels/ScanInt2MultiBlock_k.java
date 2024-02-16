package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class ScanInt2MultiBlock_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.scan_int2_array;
    private static final GPU.Kernel kernel = GPU.Kernel.scan_int2_multi_block;

    public enum Args
    {
        data,
        buffer,
        part,
        n;
    }

    public ScanInt2MultiBlock_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
