package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.kernels.GPUKernel;

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

    public MergeEntityBone_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}