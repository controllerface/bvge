package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateEntityBone_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_entity_bone, Args.class);

    public enum Args implements KernelArg
    {
        entity_bones                (Type.buffer_float16),
        entity_bone_reference_ids   (Type.buffer_int),
        entity_bone_parent_ids      (Type.buffer_int),
        target                      (Type.arg_int),
        new_armature_bone           (Type.arg_float16),
        new_armature_bone_reference (Type.arg_int),
        new_armature_bone_parent_id (Type.arg_int),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateEntityBone_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
