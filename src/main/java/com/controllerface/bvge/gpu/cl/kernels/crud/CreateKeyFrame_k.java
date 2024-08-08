package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.Kernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;

public class CreateKeyFrame_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_keyframe, Args.class);

    public enum Args implements KernelArg
    {
        key_frames     (CL_DataTypes.cl_float4.buffer_name()),
        frame_times    (CL_DataTypes.cl_float.buffer_name()),
        target         (CL_DataTypes.cl_int.name()),
        new_keyframe   (CL_DataTypes.cl_float4.name()),
        new_frame_time (CL_DataTypes.cl_float.name()),

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
