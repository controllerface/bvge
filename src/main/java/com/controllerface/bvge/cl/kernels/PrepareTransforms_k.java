package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class PrepareTransforms_k extends GPUKernel
{
    public enum Args
    {
        transforms,
        hull_rotations,
        indices,
        transforms_out,
        offset;
    }

    public PrepareTransforms_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
