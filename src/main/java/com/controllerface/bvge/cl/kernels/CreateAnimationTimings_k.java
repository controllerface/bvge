package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CreateAnimationTimings_k extends GPUKernel
{
    public enum Args
    {
        animation_timings,
        target,
        new_animation_timing;
    }

    public CreateAnimationTimings_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
