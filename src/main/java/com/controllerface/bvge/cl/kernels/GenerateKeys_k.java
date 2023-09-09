package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class GenerateKeys_k extends GPUKernel
{
    public GenerateKeys_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.generate_keys), 7);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
        def_arg(2, Sizeof.cl_mem);
        def_arg(3, Sizeof.cl_mem);
        def_arg(4, Sizeof.cl_int);
        def_arg(5, Sizeof.cl_int);
        def_arg(6, Sizeof.cl_int);
    }

    public void set_aabb_index(Pointer aabb_index)
    {
        new_arg(0, Sizeof.cl_mem, aabb_index);
    }

    public void set_aabb_key_table(Pointer aabb_key_table)
    {
        new_arg(1, Sizeof.cl_mem, aabb_key_table);
    }
}
