package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class MoveArmatures_k extends GPUKernel
{
    public MoveArmatures_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.move_armatures), 6);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
        def_arg(2, Sizeof.cl_mem);
        def_arg(3, Sizeof.cl_mem);
        def_arg(4, Sizeof.cl_mem);
        def_arg(5, Sizeof.cl_mem);
    }

    public void set_hulls(Pointer hulls)
    {
        new_arg(0, Sizeof.cl_mem, hulls);
    }

    public void set_armatures(Pointer armatures)
    {
        new_arg(1, Sizeof.cl_mem, armatures);
    }

    public void set_hull_tables(Pointer hull_tables)
    {
        new_arg(2, Sizeof.cl_mem, hull_tables);
    }

    public void set_element_tables(Pointer element_tables)
    {
        new_arg(3, Sizeof.cl_mem, element_tables);
    }

    public void set_hull_flags(Pointer hull_flags)
    {
        new_arg(4, Sizeof.cl_mem, hull_flags);
    }

    public void set_points(Pointer points)
    {
        new_arg(5, Sizeof.cl_mem, points);
    }
}
