package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

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

    public GPUKernel init()
    {
        return this.buf_arg(Args.entities, GPU.memory.get_buffer(ENTITY))
            .buf_arg(Args.entity_flags, GPU.memory.get_buffer(ENTITY_FLAG))
            .buf_arg(Args.entity_root_hulls, GPU.memory.get_buffer(ENTITY_ROOT_HULL))
            .buf_arg(Args.entity_accel, GPU.memory.get_buffer(ENTITY_ACCEL))
            .buf_arg(Args.hull_flags, GPU.memory.get_buffer(HULL_FLAG));
    }
}
