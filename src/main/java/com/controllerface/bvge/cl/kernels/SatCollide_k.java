package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class SatCollide_k extends GPUKernel
{
    public SatCollide_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.sat_collide), 12);
        int arg_index = 0;
        def_arg(arg_index++, Sizeof.cl_mem);
        def_arg(arg_index++, Sizeof.cl_mem);
        def_arg(arg_index++, Sizeof.cl_mem);
        def_arg(arg_index++, Sizeof.cl_mem);
        def_arg(arg_index++, Sizeof.cl_mem);
        def_arg(arg_index++, Sizeof.cl_mem);
        def_arg(arg_index++, Sizeof.cl_mem);
        def_arg(arg_index++, Sizeof.cl_mem);
        def_arg(arg_index++, Sizeof.cl_mem);
        def_arg(arg_index++, Sizeof.cl_mem);
        def_arg(arg_index++, Sizeof.cl_mem);
        def_arg(arg_index++, Sizeof.cl_mem);
        System.out.printf("set %d args for %s\n", arg_index, this.getClass().getSimpleName());
    }

    public void set_hulls(Pointer hulls)
    {
        new_arg(1, Sizeof.cl_mem, hulls);
    }

    public void set_element_tables(Pointer element_tables)
    {
        new_arg(2, Sizeof.cl_mem, element_tables);
    }

    public void set_hull_flags(Pointer hull_flags)
    {
        new_arg(3, Sizeof.cl_mem, hull_flags);
    }

    public void set_vertex_tables(Pointer vertex_tables)
    {
        new_arg(4, Sizeof.cl_mem, vertex_tables);
    }

    public void set_points(Pointer points)
    {
        new_arg(5, Sizeof.cl_mem, points);
    }

    public void set_edges(Pointer edges)
    {
        new_arg(6, Sizeof.cl_mem, edges);
    }

    public void set_reactions(Pointer reactions)
    {
        new_arg(9, Sizeof.cl_mem, reactions);
    }

    public void set_masses(Pointer masses)
    {
        new_arg(10, Sizeof.cl_mem, masses);
    }
}
