package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CompactArmatureBones_k extends GPUKernel
{
    public enum Args
    {
        armature_bone_shift,
        armature_bones,
        armature_bone_tables;
    }

    public CompactArmatureBones_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
