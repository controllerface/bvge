package com.controllerface.bvge.cl.kernels;

public class ScanIntMultiBlockOut_k extends GPUKernel
{
    public enum Args
    {
        input,
        output,
        buffer,
        part,
        n;
    }

    public ScanIntMultiBlockOut_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
