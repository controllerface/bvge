package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class VerifyBufferTransfer_k extends GPUKernel
{
    public enum Args
    {
        input,
        output
    }

    public VerifyBufferTransfer_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
