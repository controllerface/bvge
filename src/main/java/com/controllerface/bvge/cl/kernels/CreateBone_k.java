package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateBone_k extends GPUKernel<CreateBone_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        bones(Sizeof.cl_mem),
        bone_index_tables(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_bone(Sizeof.cl_float16),
        new_bone_table(Sizeof.cl_int2);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CreateBone_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.gpu_crud.gpu.kernels().get(GPU.Kernel.create_bone), Args.values());
    }
}
