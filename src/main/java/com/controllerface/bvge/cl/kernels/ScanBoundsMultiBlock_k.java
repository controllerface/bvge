package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class ScanBoundsMultiBlock_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.scan_key_bank;
    private static final GPU.Kernel kernel = GPU.Kernel.scan_bounds_multi_block;

    public enum Args
    {
        bounds_bank_data,
        buffer,
        part,
        n;
    }

    public ScanBoundsMultiBlock_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
