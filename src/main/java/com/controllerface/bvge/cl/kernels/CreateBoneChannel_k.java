package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateBoneChannel_k extends GPUKernel<CreateBoneChannel_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        bone_pos_channel_tables(Sizeof.cl_mem),
        bone_rot_channel_tables(Sizeof.cl_mem),
        bone_scl_channel_tables(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_bone_pos_channel_table(Sizeof.cl_int2),
        new_bone_rot_channel_table(Sizeof.cl_int2),
        new_bone_scl_channel_table(Sizeof.cl_int2);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CreateBoneChannel_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.gpu_crud.gpu.kernels().get(GPU.Kernel.create_bone_channel), Args.values());
    }
}
