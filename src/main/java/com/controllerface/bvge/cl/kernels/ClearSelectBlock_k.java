package com.controllerface.bvge.cl.kernels;

public class ClearSelectBlock_k extends GPUKernel
{
    public enum Args
    {
        entity_flags,
        target,
    }

    public ClearSelectBlock_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
