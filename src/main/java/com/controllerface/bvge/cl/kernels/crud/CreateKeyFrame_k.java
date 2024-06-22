package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateKeyFrame_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_k_src(Kernel.create_keyframe, Args.class);

    public enum Args implements KernelArg
    {
        key_frames     (Type.float4_buffer),
        frame_times    (Type.float_buffer),
        target         (Type.int_arg),
        new_keyframe   (Type.float4_arg),
        new_frame_time (Type.float_arg),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateKeyFrame_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
