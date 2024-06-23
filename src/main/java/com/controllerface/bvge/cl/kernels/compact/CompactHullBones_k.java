package com.controllerface.bvge.cl.kernels.compact;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.crud.KernelArg;

public class CompactHullBones_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.compact_k_src(Kernel.compact_hull_bones, Args.class);

    public enum Args implements KernelArg
    {
        hull_bone_shift            (Type.buffer_int),
        hull_bones                 (Type.buffer_float16),
        hull_bind_pose_indices     (Type.buffer_int),
        hull_inv_bind_pose_indices (Type.buffer_int),

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
