package com.controllerface.bvge.cl.kernels;

public class CompleteBoundsMultiBlock_k extends GPUKernel
{
    public enum Args
    {
        bounds_bank_data,
        sz,
        buffer,
        part,
        n;
    }

    public CompleteBoundsMultiBlock_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
