package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class MergeEntityBone_k extends GPUKernel
{
    public enum Args
    {
        armature_bones_in,
        armature_bone_reference_ids_in,
        armature_bone_parent_ids_in,
        armature_bones_out,
        armature_bone_reference_ids_out,
        armature_bone_parent_ids_out,
        armature_bone_offset,
        max_entity_bone,
    }

    public MergeEntityBone_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.merge_entity_bone));
    }

    public GPUKernel init(GPUCoreMemory core_memory, CoreBufferGroup core_buffers)
    {
        return this.buf_arg(Args.armature_bones_in, core_buffers.buffer(ENTITY_BONE))
            .buf_arg(Args.armature_bone_reference_ids_in, core_buffers.buffer(ENTITY_BONE_REFERENCE_ID))
            .buf_arg(Args.armature_bone_parent_ids_in, core_buffers.buffer(ENTITY_BONE_PARENT_ID))
            .buf_arg(Args.armature_bones_out, core_memory.get_buffer(ENTITY_BONE))
            .buf_arg(Args.armature_bone_reference_ids_out, core_memory.get_buffer(ENTITY_BONE_REFERENCE_ID))
            .buf_arg(Args.armature_bone_parent_ids_out, core_memory.get_buffer(ENTITY_BONE_PARENT_ID));
    }
}
