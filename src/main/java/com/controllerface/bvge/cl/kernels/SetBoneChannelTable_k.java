package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class SetBoneChannelTable_k extends GPUKernel
{
    public enum Args
    {
        bone_channel_tables,
        target,
        new_bone_channel_table;
    }

    public SetBoneChannelTable_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
