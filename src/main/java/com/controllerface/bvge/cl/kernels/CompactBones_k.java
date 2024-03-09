package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CompactBones_k extends GPUKernel
{
    public enum Args
    {
        bone_shift,
        bone_instances,
        bone_index_tables;
    }

    public CompactBones_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
