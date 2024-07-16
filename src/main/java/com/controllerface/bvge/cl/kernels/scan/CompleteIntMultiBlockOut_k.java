package com.controllerface.bvge.cl.kernels.scan;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class CompleteIntMultiBlockOut_k extends GPUKernel
{
    public enum Args
    {
        output,
        buffer,
        part,
        n;
    }

    public CompleteIntMultiBlockOut_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
