package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateArmature_k extends GPUKernel
{
    public CreateArmature_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.create_armature), 9);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
        def_arg(2, Sizeof.cl_mem);
        def_arg(3, Sizeof.cl_mem);
        def_arg(4, Sizeof.cl_int);
        def_arg(5, Sizeof.cl_float4);
        def_arg(6, Sizeof.cl_int4);
        def_arg(7, Sizeof.cl_int2);
        def_arg(8, Sizeof.cl_float);
    }

    public void set_armatures(Pointer armatures)
    {
        new_arg(0, Sizeof.cl_mem, armatures);
    }

    public void set_armature_flags(Pointer armature_flags)
    {
        new_arg(1, Sizeof.cl_mem, armature_flags);
    }

    public void set_hull_table(Pointer hull_table)
    {
        new_arg(2, Sizeof.cl_mem, hull_table);
    }

    public void set_armature_mass(Pointer armature_mass)
    {
        new_arg(3, Sizeof.cl_mem, armature_mass);
    }
}
