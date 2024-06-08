package com.controllerface.bvge.cl.kernels;

public class CreateEntityBone_k extends GPUKernel
{
    public enum Args
    {
        entity_bones,
        entity_bone_reference_ids,
        entity_bone_parent_ids,
        target,
        new_armature_bone,
        new_armature_bone_reference,
        new_armature_bone_parent_id,
    }

    public CreateEntityBone_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
