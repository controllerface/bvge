package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

import static com.controllerface.bvge.cl.CLData.cl_float16;
import static com.controllerface.bvge.cl.CLData.cl_int;

public class CreateModelTransform_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_model_transform, Args.class);

    public enum Args implements KernelArg
    {
        model_transforms    (cl_float16.buffer_name()),
        target              (cl_int.name()),
        new_model_transform (cl_float16.name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateModelTransform_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
