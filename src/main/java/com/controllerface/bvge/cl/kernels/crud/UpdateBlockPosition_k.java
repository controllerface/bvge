package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class UpdateBlockPosition_k extends GPUKernel
{
    public enum Args
    {
        entities,
        entity_root_hulls,
        hull_point_tables,
        points,
        target,
        new_value,
    }

    public UpdateBlockPosition_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
