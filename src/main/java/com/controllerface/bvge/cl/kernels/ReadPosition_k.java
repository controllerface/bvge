package com.controllerface.bvge.cl.kernels;

public class ReadPosition_k extends GPUKernel
{
    public enum Args
    {
        entities,
        output,
        target;
    }

    public ReadPosition_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
