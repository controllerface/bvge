package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class AnimateHulls_k extends GPUKernel
{
    public AnimateHulls_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.animate_hulls), 7);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
        def_arg(2, Sizeof.cl_mem);
        def_arg(3, Sizeof.cl_mem);
        def_arg(4, Sizeof.cl_mem);
        def_arg(5, Sizeof.cl_mem);
        def_arg(6, Sizeof.cl_mem);
    }

    public void set_points(Pointer points)
    {
        new_arg(0, Sizeof.cl_mem, points);
    }

    public void set_hulls(Pointer hulls)
    {
        new_arg(1, Sizeof.cl_mem, hulls);
    }

    public void set_hull_flags(Pointer hull_flags)
    {
        new_arg(2, Sizeof.cl_mem, hull_flags);
    }

    public void set_vertex_table(Pointer vertex_table)
    {
        new_arg(3, Sizeof.cl_mem, vertex_table);
    }

    public void set_armatures(Pointer armatures)
    {
        new_arg(4, Sizeof.cl_mem, armatures);
    }

    public void set_vertex_refs(Pointer vertex_refs)
    {
        new_arg(5, Sizeof.cl_mem, vertex_refs);
    }

    public void set_bone_instances(Pointer bone_instances)
    {
        new_arg(6, Sizeof.cl_mem, bone_instances);
    }

}
