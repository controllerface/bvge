package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

import static com.controllerface.bvge.cl.CLData.cl_float16;
import static com.controllerface.bvge.cl.CLData.cl_int;

public class CreateHullBone_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_hull_bone, Args.class);

    public enum Args implements KernelArg
    {
        hull_bones                  (cl_float16.buffer_name()),
        hull_bind_pose_indicies     (cl_int.buffer_name()),
        hull_inv_bind_pose_indicies (cl_int.buffer_name()),
        target                      (cl_int.name()),
        new_hull_bone               (cl_float16.name()),
        new_hull_bind_pose_id       (cl_int.name()),
        new_hull_inv_bind_pose_id   (cl_int.name()),

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
