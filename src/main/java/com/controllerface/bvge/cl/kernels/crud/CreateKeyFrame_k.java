package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateKeyFrame_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_keyframe, Args.class);

    public enum Args implements KernelArg
    {
        key_frames     (Type.buffer_float4),
        frame_times    (Type.buffer_float),
        target         (Type.arg_int),
        new_keyframe   (Type.arg_float4),
        new_frame_time (Type.arg_float),

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
