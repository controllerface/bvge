package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CreateArmatureBone_k extends GPUKernel
{
    public enum Args
    {
        armature_bones,
        bone_bind_tables,
        target,
        new_armature_bone,
        new_bone_bind_table;
    }

    public CreateArmatureBone_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
