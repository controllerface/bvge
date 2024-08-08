package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

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
