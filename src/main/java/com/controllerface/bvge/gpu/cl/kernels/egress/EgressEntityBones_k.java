package com.controllerface.bvge.gpu.cl.kernels.egress;

import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.groups.UnorderedCoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class EgressEntityBones_k extends GPUKernel
{
    public enum Args
    {
        entity_bones_in,
        entity_bone_reference_ids_in,
        entity_bone_parent_ids_in,
        entity_bones_out,
        entity_bone_reference_ids_out,
        entity_bone_parent_ids_out,
        new_entity_bones,
        max_entity_bone,
    }

    public EgressEntityBones_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.egress_entity_bones));
    }

    public GPUKernel init(GPUCoreMemory core_memory,
                          UnorderedCoreBufferGroup sector_buffers,
                          ResizableBuffer b_entity_bone_shift)
    {
        return this.buf_arg(Args.entity_bones_in, core_memory.get_buffer(ENTITY_BONE))
            .buf_arg(Args.entity_bone_reference_ids_in, core_memory.get_buffer(ENTITY_BONE_REFERENCE_ID))
            .buf_arg(Args.entity_bone_parent_ids_in, core_memory.get_buffer(ENTITY_BONE_PARENT_ID))
            .buf_arg(Args.entity_bones_out, sector_buffers.buffer(ENTITY_BONE))
            .buf_arg(Args.entity_bone_reference_ids_out, sector_buffers.buffer(ENTITY_BONE_REFERENCE_ID))
            .buf_arg(Args.entity_bone_parent_ids_out, sector_buffers.buffer(ENTITY_BONE_PARENT_ID))
            .buf_arg(Args.new_entity_bones, b_entity_bone_shift);
    }
}
