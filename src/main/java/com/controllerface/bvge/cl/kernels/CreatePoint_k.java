package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreatePoint_k extends GPUKernel<CreatePoint_k.Args>
{
    private static final GPU.Program program = GPU.Program.gpu_crud;
    private static final GPU.Kernel kernel = GPU.Kernel.create_point;

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
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
