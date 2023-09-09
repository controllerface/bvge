package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateHull_k extends GPUKernel
{
    public CreateHull_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.create_hull), 9);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
        def_arg(2, Sizeof.cl_mem);
        def_arg(3, Sizeof.cl_mem);
        def_arg(4, Sizeof.cl_int);
        def_arg(5, Sizeof.cl_float4);
        def_arg(6, Sizeof.cl_float2);
        def_arg(7, Sizeof.cl_int4);
        def_arg(8, Sizeof.cl_int2);
    }

    public void set_hulls(Pointer hulls)
    {
        new_arg(0, Sizeof.cl_mem, hulls);
    }

    public void set_hull_rotations(Pointer hull_rotations)
    {
        new_arg(1, Sizeof.cl_mem, hull_rotations);
    }

    public void set_element_table(Pointer element_table)
    {
        new_arg(2, Sizeof.cl_mem, element_table);
    }

    public void set_hull_flags(Pointer hull_flags)
    {
        new_arg(3, Sizeof.cl_mem, hull_flags);
    }
}
