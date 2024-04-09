package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class IntegrateArmatures_k extends GPUKernel
{
    public enum Args
    {
        armatures,
        armature_root_hulls,
        armature_accel,
        hull_flags,
        args,
    }

    public IntegrateArmatures_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
