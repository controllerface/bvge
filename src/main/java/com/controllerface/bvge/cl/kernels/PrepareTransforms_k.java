package com.controllerface.bvge.cl.kernels;

public class PrepareTransforms_k extends GPUKernel
{
    public enum Args
    {
        hull_positions,
        hull_scales,
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
