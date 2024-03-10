package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CreateBone_k extends GPUKernel
{
    public enum Args
    {
        bones,
        bone_index_tables,
        target,
        new_bone,
        new_bone_table;
    }

    public CreateBone_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
