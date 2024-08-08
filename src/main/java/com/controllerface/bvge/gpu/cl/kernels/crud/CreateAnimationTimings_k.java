package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;

import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.cl_float;
import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.cl_int;

public class CreateAnimationTimings_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_animation_timings, Args.class);

    public enum Args implements KernelArg
    {
        animation_durations     (cl_float.buffer_name()),
        animation_tick_rates    (cl_float.buffer_name()),
        target                  (cl_int.name()),
        new_animation_duration  (cl_float.name()),
        new_animation_tick_rate (cl_float.name()),

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
