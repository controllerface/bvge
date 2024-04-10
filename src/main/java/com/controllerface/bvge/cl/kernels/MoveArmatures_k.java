package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class MoveArmatures_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        armatures,
        armature_flags,
        hull_tables,
        hull_point_tables,
        hull_flags,
        point_flags,
        points
    }

    public MoveArmatures_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
