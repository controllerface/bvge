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
        def_arg(arg_index++, Sizeof.cl_mem);    // __global float4 *points,
        def_arg(arg_index++, Sizeof.cl_mem);    // __global int2 *vertex_tables,
        def_arg(arg_index++, Sizeof.cl_mem);    // __global int4 *bone_tables,
        def_arg(arg_index++, Sizeof.cl_int);    // int target,
        def_arg(arg_index++, Sizeof.cl_float4); // float4 new_point,
        def_arg(arg_index++, Sizeof.cl_int2);   // int2 new_vertex_table,
        def_arg(arg_index++, Sizeof.cl_int4);   // int4 new_bone_table
        System.out.printf("set %d args for %s\n", arg_index, this.getClass().getSimpleName());
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
