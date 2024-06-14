package com.controllerface.bvge.cl.kernels;

public class CompleteIntMultiBlock_k extends GPUKernel
{
    public enum Args
    {
        data,
        buffer,
        part,
        n;
    }

    public CompleteIntMultiBlock_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
