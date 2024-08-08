package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

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

    public IntegrateEntities_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.integrate_entities));
    }
}
