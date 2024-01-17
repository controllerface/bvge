package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateTextureUV_k extends GPUKernel
{
    public CreateTextureUV_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.create_texture_uv), 3);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_int);
        def_arg(2, Sizeof.cl_float2);
    }

    public void set_texture_uvs(Pointer texture_uvs)
    {
        new_arg(0, Sizeof.cl_mem, texture_uvs);
    }
}
