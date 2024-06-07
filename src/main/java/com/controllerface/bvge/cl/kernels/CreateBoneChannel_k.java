package com.controllerface.bvge.cl.kernels;

public class CreateBoneChannel_k extends GPUKernel
{
    public enum Args
    {
        animation_timing_indices,
        bone_pos_channel_tables,
        bone_rot_channel_tables,
        bone_scl_channel_tables,
        target,
        new_animation_timing_index,
        new_bone_pos_channel_table,
        new_bone_rot_channel_table,
        new_bone_scl_channel_table;
    }

    public CreateBoneChannel_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
