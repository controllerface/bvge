package com.controllerface.bvge.cl.kernels;

public class ScanIntSingleBlock_k extends GPUKernel
{
    public enum Args
    {
        data,
        buffer,
        n;
    }

    public ScanIntSingleBlock_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
