package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

import static com.controllerface.bvge.cl.CLData.cl_int;
import static com.controllerface.bvge.cl.CLData.cl_int2;

public class SetBoneChannelTable_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.set_bone_channel_table, Args.class);

    public enum Args implements KernelArg
    {
        bone_channel_tables    (cl_int2.buffer_name()),
        target                 (cl_int.name()),
        new_bone_channel_table (cl_int2.name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public SetBoneChannelTable_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
