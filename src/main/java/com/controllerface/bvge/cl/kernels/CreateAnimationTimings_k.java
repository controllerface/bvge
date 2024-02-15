package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateAnimationTimings_k extends GPUKernel<CreateAnimationTimings_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        animation_timings(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_animation_timing(Sizeof.cl_double2);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CreateAnimationTimings_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.gpu_crud.gpu.kernels().get(GPU.Kernel.create_animation_timings), Args.values());
    }
}
