package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.KernelArg;

import static com.controllerface.bvge.cl.CLData.cl_float2;
import static com.controllerface.bvge.cl.CLData.cl_int;

public class CreateTextureUV_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_texture_uv, Args.class);

    public enum Args implements KernelArg
    {
        texture_uvs    (cl_float2.buffer_name()),
        target         (cl_int.name()),
        new_texture_uv (cl_float2.name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateTextureUV_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
