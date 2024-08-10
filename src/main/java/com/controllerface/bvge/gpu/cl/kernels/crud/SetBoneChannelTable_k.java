package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.ReferenceBufferGroup;

import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.cl_int;
import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.cl_int2;
import static com.controllerface.bvge.memory.types.ReferenceBufferType.BONE_ANIM_CHANNEL_TABLE;

public class SetBoneChannelTable_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.set_bone_channel_table, Args.class);

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

    public SetBoneChannelTable_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.set_bone_channel_table));
    }

    public GPUKernel init(ReferenceBufferGroup reference_buffers)
    {
        return this.buf_arg(Args.bone_channel_tables, reference_buffers.buffer(BONE_ANIM_CHANNEL_TABLE));
    }
}
