package com.controllerface.bvge.cl.kernels;

public class ScanIntSingleBlockOut_k extends GPUKernel
{
    public enum Args
    {
        input,
        output,
        buffer,
        n;
    }

    public ScanIntSingleBlockOut_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
