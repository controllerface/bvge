package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CreateBoneChannel_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.gpu_crud;
    private static final GPU.Kernel kernel = GPU.Kernel.create_bone_channel;

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

    public CreateBoneChannel_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
