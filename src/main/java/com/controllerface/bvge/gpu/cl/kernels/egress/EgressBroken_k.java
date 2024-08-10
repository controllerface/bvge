package com.controllerface.bvge.gpu.cl.kernels.egress;

import com.controllerface.bvge.gpu.cl.buffers.BufferGroup;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.types.BrokenBufferType;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

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

    public GPUKernel init(GPUCoreMemory core_memory, BufferGroup<BrokenBufferType> broken_group, CL_Buffer egress_size)
    {
        return this.buf_arg(Args.entities, core_memory.get_buffer(ENTITY))
            .buf_arg(Args.entity_flags, core_memory.get_buffer(ENTITY_FLAG))
            .buf_arg(Args.entity_types, core_memory.get_buffer(ENTITY_TYPE))
            .buf_arg(Args.entity_model_ids, core_memory.get_buffer(ENTITY_MODEL_ID))
            .buf_arg(Args.positions, broken_group.buffer(BrokenBufferType.BROKEN_POSITIONS))
            .buf_arg(Args.types, broken_group.buffer(BrokenBufferType.BROKEN_ENTITY_TYPES))
            .buf_arg(Args.model_ids, broken_group.buffer(BrokenBufferType.BROKEN_MODEL_IDS))
            .buf_arg(Args.counter, egress_size);
    }
}
