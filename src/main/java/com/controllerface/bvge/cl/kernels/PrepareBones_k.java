package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class PrepareBones_k extends GPUKernel
{
    public PrepareBones_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.prepare_bones), 8);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
        def_arg(2, Sizeof.cl_mem);
        def_arg(3, Sizeof.cl_mem);
        def_arg(4, Sizeof.cl_mem);
        def_arg(5, Sizeof.cl_mem);
        def_arg(6, Sizeof.cl_mem);
        def_arg(7, Sizeof.cl_int);
    }

    public void set_bone_instances(Pointer bone_instances)
    {
        new_arg(0, Sizeof.cl_mem, bone_instances);
    }

    public void set_bone_references(Pointer bone_references)
    {
        new_arg(1, Sizeof.cl_mem, bone_references);
    }

    public void set_bone_index_tables(Pointer bone_index_table)
    {
        new_arg(2, Sizeof.cl_mem, bone_index_table);
    }

    public void set_hulls(Pointer hulls)
    {
        new_arg(3, Sizeof.cl_mem, hulls);
    }

    public void set_armatures(Pointer armatures)
    {
        new_arg(4, Sizeof.cl_mem, armatures);
    }

    public void set_hull_flags(Pointer hull_flags)
    {
        new_arg(5, Sizeof.cl_mem, hull_flags);
    }
}
