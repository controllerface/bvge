package com.controllerface.bvge.gpu.cl.kernels.compact;

import com.controllerface.bvge.gpu.cl.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.Kernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class CompactHullBones_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.compact_k_src(Kernel.compact_hull_bones, Args.class);

    public enum Args implements KernelArg
    {
        hull_bone_shift            (CL_DataTypes.cl_int.buffer_name()),
        hull_bones                 (HULL_BONE.data_type().buffer_name()),
        hull_bind_pose_indices     (HULL_BONE_BIND_POSE.data_type().buffer_name()),
        hull_inv_bind_pose_indices (HULL_BONE_INV_BIND_POSE.data_type().buffer_name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CompactHullBones_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
