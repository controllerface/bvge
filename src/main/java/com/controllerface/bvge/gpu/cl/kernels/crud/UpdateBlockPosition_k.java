package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

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

    public UpdateBlockPosition_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
