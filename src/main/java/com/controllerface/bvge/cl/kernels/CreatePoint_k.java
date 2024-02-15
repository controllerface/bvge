package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreatePoint_k extends GPUKernel<CreatePoint_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        points(Sizeof.cl_mem),
        vertex_tables(Sizeof.cl_mem),
        bone_tables(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_point(Sizeof.cl_float4),
        new_vertex_table(Sizeof.cl_int4),
        new_bone_table(Sizeof.cl_int4);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CreatePoint_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.gpu_crud.gpu.kernels().get(GPU.Kernel.create_point), Args.values());
    }
}
