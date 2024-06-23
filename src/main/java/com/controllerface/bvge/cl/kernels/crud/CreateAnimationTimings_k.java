package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateAnimationTimings_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_animation_timings, Args.class);

    public enum Args implements KernelArg
    {
        animation_durations     (Type.buffer_float),
        animation_tick_rates    (Type.buffer_float),
        target                  (Type.arg_int),
        new_animation_duration  (Type.arg_float),
        new_animation_tick_rate (Type.arg_float),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateAnimationTimings_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }

}
