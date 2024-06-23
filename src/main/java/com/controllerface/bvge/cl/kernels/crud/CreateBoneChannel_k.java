package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateBoneChannel_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_bone_channel, Args.class);;

    public enum Args implements KernelArg
    {
        animation_timing_indices   (Type.buffer_int),
        bone_pos_channel_tables    (Type.buffer_int2),
        bone_rot_channel_tables    (Type.buffer_int2),
        bone_scl_channel_tables    (Type.buffer_int2),
        target                     (Type.arg_int),
        new_animation_timing_index (Type.arg_int),
        new_bone_pos_channel_table (Type.arg_int2),
        new_bone_rot_channel_table (Type.arg_int2),
        new_bone_scl_channel_table (Type.arg_int2),

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
