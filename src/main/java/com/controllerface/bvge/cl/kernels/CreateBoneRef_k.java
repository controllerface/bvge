package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateBoneRef_k extends GPUKernel
{
    public CreateBoneRef_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.create_bone_reference), 3);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_int);
        def_arg(2, Sizeof.cl_float16);
    }

    public void set_bone_refs(Pointer bone_ref)
    {
        new_arg(0, Sizeof.cl_mem, bone_ref);
    }

}
