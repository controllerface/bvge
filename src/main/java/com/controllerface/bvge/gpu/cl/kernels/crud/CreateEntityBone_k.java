package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class CreateEntityBone_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_entity_bone, Args.class);

    public enum Args implements KernelArg
    {
        entity_bones                (CL_DataTypes.cl_float16.buffer_name()),
        entity_bone_reference_ids   (CL_DataTypes.cl_int.buffer_name()),
        entity_bone_parent_ids      (CL_DataTypes.cl_int.buffer_name()),
        target                      (CL_DataTypes.cl_int.name()),
        new_armature_bone           (CL_DataTypes.cl_float16.name()),
        new_armature_bone_reference (CL_DataTypes.cl_int.name()),
        new_armature_bone_parent_id (CL_DataTypes.cl_int.name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateEntityBone_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.create_entity_bone));
    }

    public GPUKernel init(CoreBufferGroup core_buffers)
    {
        return this.buf_arg(Args.entity_bones, core_buffers.buffer(ENTITY_BONE))
            .buf_arg(Args.entity_bone_reference_ids, core_buffers.buffer(ENTITY_BONE_REFERENCE_ID))
            .buf_arg(Args.entity_bone_parent_ids, core_buffers.buffer(ENTITY_BONE_PARENT_ID));
    }
}
