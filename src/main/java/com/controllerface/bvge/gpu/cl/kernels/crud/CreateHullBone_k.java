package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;

public class CreateHullBone_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_hull_bone, Args.class);

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

    public CreateHullBone_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
