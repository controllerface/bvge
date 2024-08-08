package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.Kernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;

public class CreateHullBone_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_hull_bone, Args.class);

    public enum Args implements KernelArg
    {
        hull_bones                  (CL_DataTypes.cl_float16.buffer_name()),
        hull_bind_pose_indicies     (CL_DataTypes.cl_int.buffer_name()),
        hull_inv_bind_pose_indicies (CL_DataTypes.cl_int.buffer_name()),
        target                      (CL_DataTypes.cl_int.name()),
        new_hull_bone               (CL_DataTypes.cl_float16.name()),
        new_hull_bind_pose_id       (CL_DataTypes.cl_int.name()),
        new_hull_inv_bind_pose_id   (CL_DataTypes.cl_int.name()),

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
