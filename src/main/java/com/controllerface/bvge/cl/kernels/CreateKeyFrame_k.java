package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateKeyFrame_k extends GPUKernel<CreateKeyFrame_k.Args>
{
    private static final GPU.Program program = GPU.Program.gpu_crud;
    private static final GPU.Kernel kernel = GPU.Kernel.create_keyframe;

    public enum Args implements GPUKernelArg
    {
        key_frames(Sizeof.cl_mem),
        frame_times(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_keyframe(Sizeof.cl_float4),
        new_frame_time(Sizeof.cl_double);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CreateKeyFrame_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
