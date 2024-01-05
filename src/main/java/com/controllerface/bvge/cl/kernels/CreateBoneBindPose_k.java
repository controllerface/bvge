package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateBoneBindPose_k extends GPUKernel
{
    public CreateBoneBindPose_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.create_bone_bind_pose), 5);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
        def_arg(2, Sizeof.cl_int);
        def_arg(3, Sizeof.cl_float16);
        def_arg(4, Sizeof.cl_int);
    }

    public void set_bone_binds(Pointer bone_ref)
    {
        new_arg(0, Sizeof.cl_mem, bone_ref);
    }

    public void set_bone_parents(Pointer bone_parents)
    {
        new_arg(1, Sizeof.cl_mem, bone_parents);
    }
}
