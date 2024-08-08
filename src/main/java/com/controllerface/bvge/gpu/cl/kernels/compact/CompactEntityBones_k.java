package com.controllerface.bvge.gpu.cl.kernels.compact;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;

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

    public CompactEntityBones_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
