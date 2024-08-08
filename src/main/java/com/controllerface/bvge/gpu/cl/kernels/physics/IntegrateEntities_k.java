package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class IntegrateEntities_k extends GPUKernel
{
    public enum Args
    {
        entities,
        entity_flags,
        entity_root_hulls,
        entity_accel,
        hull_flags,
        args,
        max_entity,
    }

    public IntegrateEntities_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
