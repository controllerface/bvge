package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateBoneChannel_k extends GPUKernel<CreateBoneChannel_k.Args>
{
    private static final GPU.Program program = GPU.Program.gpu_crud;
    private static final GPU.Kernel kernel = GPU.Kernel.create_bone_channel;

    public enum Args implements GPUKernelArg
    {
        animation_timing_indices(Sizeof.cl_mem),
        bone_pos_channel_tables(Sizeof.cl_mem),
        bone_rot_channel_tables(Sizeof.cl_mem),
        bone_scl_channel_tables(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_animation_timing_index(Sizeof.cl_int),
        new_bone_pos_channel_table(Sizeof.cl_int2),
        new_bone_rot_channel_table(Sizeof.cl_int2),
        new_bone_scl_channel_table(Sizeof.cl_int2);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CreateBoneChannel_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
