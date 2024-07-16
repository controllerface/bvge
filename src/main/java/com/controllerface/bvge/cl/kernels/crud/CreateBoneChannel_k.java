package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.KernelArg;

import static com.controllerface.bvge.cl.CLData.cl_int;
import static com.controllerface.bvge.cl.CLData.cl_int2;

public class CreateBoneChannel_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_bone_channel, Args.class);;

    public enum Args implements KernelArg
    {
        animation_timing_indices   (cl_int.buffer_name()),
        bone_pos_channel_tables    (cl_int2.buffer_name()),
        bone_rot_channel_tables    (cl_int2.buffer_name()),
        bone_scl_channel_tables    (cl_int2.buffer_name()),
        target                     (cl_int.name()),
        new_animation_timing_index (cl_int.name()),
        new_bone_pos_channel_table (cl_int2.name()),
        new_bone_rot_channel_table (cl_int2.name()),
        new_bone_scl_channel_table (cl_int2.name()),

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
