package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateHullBone_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_hull_bone, Args.class);

    public enum Args implements KernelArg
    {
        hull_bones                  (Type.buffer_float16),
        hull_bind_pose_indicies     (Type.buffer_int),
        hull_inv_bind_pose_indicies (Type.buffer_int),
        target                      (Type.arg_int),
        new_hull_bone               (Type.arg_float16),
        new_hull_bind_pose_id       (Type.arg_int),
        new_hull_inv_bind_pose_id   (Type.arg_int),

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
