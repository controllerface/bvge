package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateKeyFrame_k extends GPUKernel<CreateKeyFrame_k.Args>
{
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

    public CreateKeyFrame_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.gpu_crud.gpu.kernels().get(GPU.Kernel.create_keyframe), Args.values());
    }
}
