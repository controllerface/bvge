package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;

import static com.controllerface.bvge.gpu.cl.CL_DataTypes.cl_float16;
import static com.controllerface.bvge.gpu.cl.CL_DataTypes.cl_int;

public class CreateBoneBindPose_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_bone_bind_pose, Args.class);

    public enum Args implements KernelArg
    {
        bone_bind_poses    (cl_float16.buffer_name()),
        bone_layers        (cl_int.buffer_name()),
        target             (cl_int.name()),
        new_bone_bind_pose (cl_float16.name()),
        new_bone_layer     (cl_int.name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateBoneBindPose_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
