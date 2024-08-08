package com.controllerface.bvge.gpu.cl.kernels.egress;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class EgressBroken_k extends GPUKernel
{
    public enum Args
    {
        entities,
        entity_flags,
        entity_types,
        entity_model_ids,
        positions,
        types,
        model_ids,
        counter,
        max_entity,
    }

    public EgressBroken_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
