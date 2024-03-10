package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class MoveArmatures_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        armatures,
        hull_tables,
        element_tables,
        hull_flags,
        points;
    }

    public MoveArmatures_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
