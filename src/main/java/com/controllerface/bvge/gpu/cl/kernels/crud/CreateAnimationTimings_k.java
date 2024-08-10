package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.ReferenceBufferGroup;

import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.cl_float;
import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.cl_int;
import static com.controllerface.bvge.memory.types.ReferenceBufferType.ANIM_DURATION;
import static com.controllerface.bvge.memory.types.ReferenceBufferType.ANIM_TICK_RATE;

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

    public CreateAnimationTimings_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.create_animation_timings));
    }

    public GPUKernel init(ReferenceBufferGroup reference_buffers)
    {
        return this.buf_arg(Args.animation_durations, reference_buffers.buffer(ANIM_DURATION))
            .buf_arg(Args.animation_tick_rates, reference_buffers.buffer(ANIM_TICK_RATE));
    }

}
