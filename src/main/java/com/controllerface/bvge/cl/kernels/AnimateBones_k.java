package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class AnimateBones_k extends GPUKernel
{
    public enum Args
    {
        bones,
        bone_references,
        armature_bones,
        bone_index_tables;
    }

    public AnimateBones_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
