package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class LocateOutOfBounds_k extends GPUKernel
{
    public LocateOutOfBounds_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.locate_out_of_bounds), 4);
        def_arg(0, Sizeof.cl_mem); // hull tables
        def_arg(1, Sizeof.cl_mem); // hull flags
        def_arg(2, Sizeof.cl_mem); // armature flags
        def_arg(3, Sizeof.cl_mem);
    }

    public void set_hull_tables(Pointer hull_tables)
    {
        new_arg(0, Sizeof.cl_mem, hull_tables);
    }

    public void set_hull_flags(Pointer hull_flags)
    {
        new_arg(1, Sizeof.cl_mem, hull_flags);
    }

    public void set_armature_flags(Pointer armature_flags)
    {
        new_arg(2, Sizeof.cl_mem, armature_flags);
    }
}
