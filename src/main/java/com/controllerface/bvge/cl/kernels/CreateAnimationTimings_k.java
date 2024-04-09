package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CreateAnimationTimings_k extends GPUKernel
{
    public enum Args
    {
        animation_durations,
        animation_tick_rates,
        target,
        new_animation_duration,
        new_animation_tick_rate,
    }

    public CreateAnimationTimings_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
