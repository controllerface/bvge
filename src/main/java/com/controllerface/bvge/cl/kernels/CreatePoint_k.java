package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreatePoint_k extends GPUKernel
{
    public CreatePoint_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.create_point), 7);
        int arg_index = 0;
        def_arg(arg_index++, Sizeof.cl_mem);      // points
        def_arg(arg_index++, Sizeof.cl_mem);      // vertex tables
        def_arg(arg_index++, Sizeof.cl_mem);      // bone tables
        def_arg(arg_index++, Sizeof.cl_int);      // target index
        def_arg(arg_index++, Sizeof.cl_float4);   // new point
        def_arg(arg_index++, Sizeof.cl_int2);     // new vertex table
        def_arg(arg_index++, Sizeof.cl_int4);     // new bone table
        System.out.println("Built: " + arg_index + "params" + this.getClass().getSimpleName());
    }

    public void set_points(Pointer points)
    {
        new_arg(0, Sizeof.cl_mem, points);
    }

    public void set_vertex_tables(Pointer vertex_tables)
    {
        new_arg(1, Sizeof.cl_mem, vertex_tables);
    }

    public void set_bone_tables(Pointer bone_tables)
    {
        new_arg(2, Sizeof.cl_mem, bone_tables);
    }
}
