package com.controllerface.bvge.gpu.cl.kernels.egress;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

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

    public EgressBroken_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.egress_broken));
    }
}
