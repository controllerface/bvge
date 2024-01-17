package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateVertexRef_k extends GPUKernel
{
    public CreateVertexRef_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.create_vertex_reference), 7);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
        def_arg(2, Sizeof.cl_mem);
        def_arg(3, Sizeof.cl_int);
        def_arg(4, Sizeof.cl_float2);
        def_arg(5, Sizeof.cl_float4);
        def_arg(6, Sizeof.cl_int2);
    }

    public void set_vertex_refs(Pointer vertex_refs)
    {
        new_arg(0, Sizeof.cl_mem, vertex_refs);
    }

    public void set_vertex_weights(Pointer vertex_weights)
    {
        new_arg(1, Sizeof.cl_mem, vertex_weights);
    }

    public void set_uv_tables(Pointer uv_tables)
    {
        new_arg(2, Sizeof.cl_mem, uv_tables);
    }
}
