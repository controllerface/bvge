package com.controllerface.bvge.gpu.cl.kernels.egress;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class EgressCollected_k extends GPUKernel
{
    public enum Args
    {
        entity_flags,
        entity_types,
        types,
        counter,
        max_entity,
    }

    public EgressCollected_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
