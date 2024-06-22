package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateHullBone_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_k_src(Kernel.create_hull_bone, Args.class);

    public enum Args implements KernelArg
    {
        hull_bones                  (Type.float16_buffer),
        hull_bind_pose_indicies     (Type.int_buffer),
        hull_inv_bind_pose_indicies (Type.int_buffer),
        target                      (Type.int_arg),
        new_hull_bone               (Type.float16_arg),
        new_hull_bind_pose_id       (Type.int_arg),
        new_hull_inv_bind_pose_id   (Type.int_arg),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateHullBone_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
