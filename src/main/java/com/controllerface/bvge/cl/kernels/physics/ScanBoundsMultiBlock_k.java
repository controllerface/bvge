package com.controllerface.bvge.cl.kernels.physics;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class ScanBoundsMultiBlock_k extends GPUKernel
{
    public enum Args
    {
        bounds_bank_data,
        buffer,
        part,
        n;
    }

    public ScanBoundsMultiBlock_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
