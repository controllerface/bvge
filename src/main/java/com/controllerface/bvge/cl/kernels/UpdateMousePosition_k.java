package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class UpdateMousePosition_k extends GPUKernel
{
    public enum Args
    {
        armature_root_hulls,
        hull_point_tables,
        points,
        target,
        new_value;
    }

    public UpdateMousePosition_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
