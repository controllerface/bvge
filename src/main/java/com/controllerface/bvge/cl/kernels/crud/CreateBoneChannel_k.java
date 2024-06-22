package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateBoneChannel_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_k_src(Kernel.create_bone_channel, Args.class);;

    public enum Args implements KernelArg
    {
        animation_timing_indices   (Type.int_buffer),
        bone_pos_channel_tables    (Type.int2_buffer),
        bone_rot_channel_tables    (Type.int2_buffer),
        bone_scl_channel_tables    (Type.int2_buffer),
        target                     (Type.int_arg),
        new_animation_timing_index (Type.int_arg),
        new_bone_pos_channel_table (Type.int2_arg),
        new_bone_rot_channel_table (Type.int2_arg),
        new_bone_scl_channel_table (Type.int2_arg),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateBoneChannel_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
