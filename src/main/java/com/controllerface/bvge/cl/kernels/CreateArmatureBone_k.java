package com.controllerface.bvge.cl.kernels;

public class CreateArmatureBone_k extends GPUKernel
{
    public enum Args
    {
        armature_bones,
        armature_bone_reference_ids,
        armature_bone_parent_ids,
        target,
        new_armature_bone,
        new_armature_bone_reference,
        new_armature_bone_parent_id,
    }

    public CreateArmatureBone_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
