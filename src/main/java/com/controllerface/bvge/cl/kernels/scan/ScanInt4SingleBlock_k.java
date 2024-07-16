package com.controllerface.bvge.cl.kernels.scan;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class ScanInt4SingleBlock_k extends GPUKernel
{
    public enum Args
    {
        data,
        buffer,
        n;
    }

    public ScanInt4SingleBlock_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
