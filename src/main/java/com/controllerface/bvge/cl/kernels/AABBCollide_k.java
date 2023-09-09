package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class AABBCollide_k extends GPUKernel
{
    public AABBCollide_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.aabb_collide), 14);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
        def_arg(2, Sizeof.cl_mem);
        def_arg(3, Sizeof.cl_mem);
        def_arg(4, Sizeof.cl_mem);
        def_arg(5, Sizeof.cl_mem);
        def_arg(6, Sizeof.cl_mem);
        def_arg(7, Sizeof.cl_mem);
        def_arg(8, Sizeof.cl_mem);
        def_arg(9, Sizeof.cl_mem);
        def_arg(10, Sizeof.cl_mem);
        def_arg(11, Sizeof.cl_mem);
        def_arg(12, Sizeof.cl_int);
        def_arg(13, Sizeof.cl_int);
    }

    public void set_aabb(Pointer aabb)
    {
        new_arg(0, Sizeof.cl_mem, aabb);
    }

    public void set_aabb_key_table(Pointer aabb_key_table)
    {
        new_arg(1, Sizeof.cl_mem, aabb_key_table);
    }

    public void set_hull_flags(Pointer hull_flags)
    {
        new_arg(2, Sizeof.cl_mem, hull_flags);
    }

}
