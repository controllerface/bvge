package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.ReferenceBufferGroup;

import static com.controllerface.bvge.memory.types.ReferenceBufferType.*;

public class CreateBoneChannel_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_bone_channel, Args.class);

    public enum Args implements KernelArg
    {
        animation_timing_indices   (CL_DataTypes.cl_int.buffer_name()),
        bone_pos_channel_tables    (CL_DataTypes.cl_int2.buffer_name()),
        bone_rot_channel_tables    (CL_DataTypes.cl_int2.buffer_name()),
        bone_scl_channel_tables    (CL_DataTypes.cl_int2.buffer_name()),
        target                     (CL_DataTypes.cl_int.name()),
        new_animation_timing_index (CL_DataTypes.cl_int.name()),
        new_bone_pos_channel_table (CL_DataTypes.cl_int2.name()),
        new_bone_rot_channel_table (CL_DataTypes.cl_int2.name()),
        new_bone_scl_channel_table (CL_DataTypes.cl_int2.name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateBoneChannel_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.create_bone_channel));
    }

    public GPUKernel init(ReferenceBufferGroup reference_buffers)
    {
        return this.buf_arg(Args.animation_timing_indices, reference_buffers.buffer(ANIM_TIMING_INDEX))
            .buf_arg(Args.bone_pos_channel_tables, reference_buffers.buffer(ANIM_POS_CHANNEL))
            .buf_arg(Args.bone_rot_channel_tables, reference_buffers.buffer(ANIM_ROT_CHANNEL))
            .buf_arg(Args.bone_scl_channel_tables, reference_buffers.buffer(ANIM_SCL_CHANNEL));
    }
}
