package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateMeshFace_k extends GPUKernel
{
    public CreateMeshFace_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.create_mesh_face), 3);
        int arg_index = 0;
        def_arg(arg_index++, Sizeof.cl_mem);
        def_arg(arg_index++, Sizeof.cl_int);
        def_arg(arg_index++, Sizeof.cl_float4);
        System.out.printf("set %d args for %s\n", arg_index, this.getClass().getSimpleName());
    }

    public void set_mesh_faces(Pointer mesh_faces)
    {
        new_arg(0, Sizeof.cl_mem, mesh_faces);
    }
}
