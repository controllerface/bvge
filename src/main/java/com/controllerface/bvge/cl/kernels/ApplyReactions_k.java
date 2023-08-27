package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class ApplyReactions_k extends GPUKernel
{
    public ApplyReactions_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.apply_reactions), 5);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
        def_arg(2, Sizeof.cl_mem);
        def_arg(3, Sizeof.cl_mem);
        def_arg(4, Sizeof.cl_mem);
    }

    public void set_points(Pointer points)
    {
        new_arg(1, Sizeof.cl_mem, points);
    }

    public void set_point_anti_grav(Pointer point_anti_grav)
    {
        new_arg(2, Sizeof.cl_mem, point_anti_grav);
    }

    public void set_point_reactions(Pointer point_reactions)
    {
        new_arg(3, Sizeof.cl_mem, point_reactions);
    }

    public void set_point_offsets(Pointer point_offsets)
    {
        new_arg(4, Sizeof.cl_mem, point_offsets);
    }
}
