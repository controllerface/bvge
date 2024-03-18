package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class BufferTransfer_k extends GPUKernel
{
    public enum Args
    {
        input,
        output
    }

    public BufferTransfer_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
