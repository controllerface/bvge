package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateBoneRef_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_k_src(Kernel.create_bone_reference, Args.class);;

    public enum Args implements KernelArg
    {
        bone_references    (Type.float16_buffer),
        target             (Type.int_arg),
        new_bone_reference (Type.float16_arg),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateBoneRef_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
