package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateVertexRef_k extends GPUKernel<CreateVertexRef_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        vertex_references(Sizeof.cl_mem),
        vertex_weights(Sizeof.cl_mem),
        uv_tables(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_vertex_reference(Sizeof.cl_float2),
        new_vertex_weights(Sizeof.cl_float4),
        new_uv_table(Sizeof.cl_int2);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CreateVertexRef_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.gpu_crud.gpu.kernels().get(GPU.Kernel.create_vertex_reference), Args.values());
    }
}
