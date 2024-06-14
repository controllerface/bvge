package com.controllerface.bvge.cl.kernels;

public class CompleteInt2MultiBlock_k extends GPUKernel
{
    public enum Args
    {
        data,
        buffer,
        part,
        n;
    }

    public CompleteInt2MultiBlock_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
