package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompactBones_k extends GPUKernel
{
    public CompactBones_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.compact_bones), 3);
        def_arg(0, Sizeof.cl_mem);
        def_arg(1, Sizeof.cl_mem);
        def_arg(2, Sizeof.cl_mem);
    }

    public void set_bone_shift(Pointer bone_shift)
    {
        new_arg(0, Sizeof.cl_mem, bone_shift);
    }

    public void set_bone_instances(Pointer bone_instances)
    {
        new_arg(1, Sizeof.cl_mem, bone_instances);
    }

    public void set_bone_indices(Pointer bone_indices)
    {
        new_arg(2, Sizeof.cl_mem, bone_indices);
    }
}
