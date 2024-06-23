package com.controllerface.bvge.cl.kernels;

public class CompactEntityBones_k extends GPUKernel
{
    public enum Args
    {
        armature_bone_shift,
        armature_bones,
        armature_bone_reference_ids,
        armature_bone_parent_ids,
    }

    public CompactEntityBones_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
