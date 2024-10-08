package com.controllerface.bvge.gpu.cl.kernels.compact;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class CompactEntityBones_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.compact_k_src(KernelType.compact_entity_bones, Args.class);

    public enum Args implements KernelArg
    {
        entity_bone_shift           (CL_DataTypes.cl_int.buffer_name()),
        entity_bones                (ENTITY_BONE.data_type().buffer_name()),
        entity_bone_reference_ids   (ENTITY_BONE_REFERENCE_ID.data_type().buffer_name()),
        entity_bone_parent_ids      (ENTITY_BONE_PARENT_ID.data_type().buffer_name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CompactEntityBones_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.compact_entity_bones));
    }

    public GPUKernel init(CoreBufferGroup sector_buffers, ResizableBuffer b_entity_bone_shift)
    {
        return this.buf_arg(Args.entity_bone_shift, b_entity_bone_shift)
            .buf_arg(Args.entity_bones, sector_buffers.buffer(ENTITY_BONE))
            .buf_arg(Args.entity_bone_reference_ids, sector_buffers.buffer(ENTITY_BONE_REFERENCE_ID))
            .buf_arg(Args.entity_bone_parent_ids, sector_buffers.buffer(ENTITY_BONE_PARENT_ID));
    }
}
