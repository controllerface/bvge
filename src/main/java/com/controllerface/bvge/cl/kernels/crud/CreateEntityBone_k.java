package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

import static com.controllerface.bvge.cl.CLData.cl_float16;
import static com.controllerface.bvge.cl.CLData.cl_int;

public class CreateEntityBone_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_entity_bone, Args.class);

    public enum Args implements KernelArg
    {
        entity_bones                (cl_float16.buffer_name()),
        entity_bone_reference_ids   (cl_int.buffer_name()),
        entity_bone_parent_ids      (cl_int.buffer_name()),
        target                      (cl_int.name()),
        new_armature_bone           (cl_float16.name()),
        new_armature_bone_reference (cl_int.name()),
        new_armature_bone_parent_id (cl_int.name()),

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
