package com.controllerface.bvge.gpu.cl.kernels.egress;

import com.controllerface.bvge.gpu.cl.buffers.BufferGroup;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.types.CollectedBufferType;

import static com.controllerface.bvge.memory.types.CoreBufferType.ENTITY_FLAG;
import static com.controllerface.bvge.memory.types.CoreBufferType.ENTITY_TYPE;

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

    public EgressCollected_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.egress_collected));
    }

    public GPUKernel init(GPUCoreMemory core_memory, BufferGroup<CollectedBufferType> collected_group, CL_Buffer egress_size)
    {
        return this.buf_arg(Args.entity_flags, core_memory.get_buffer(ENTITY_FLAG))
            .buf_arg(Args.entity_types, core_memory.get_buffer(ENTITY_TYPE))
            .buf_arg(Args.types, collected_group.buffer(CollectedBufferType.TYPES))
            .buf_arg(Args.counter, egress_size);
    }
}
