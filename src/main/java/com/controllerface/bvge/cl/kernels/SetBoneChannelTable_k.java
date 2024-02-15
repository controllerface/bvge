package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class SetBoneChannelTable_k extends GPUKernel<SetBoneChannelTable_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        bone_channel_tables(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_bone_channel_table(Sizeof.cl_int2);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public SetBoneChannelTable_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.gpu_crud.gpu.kernels().get(GPU.Kernel.set_bone_channel_table), Args.values());
    }
}
