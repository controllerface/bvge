package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class SortReactions_k extends GPUKernel
{
    public enum Args
    {
        reactions_in,
        reactions_out,
        reaction_index,
        point_reactions,
        point_offsets,
        max_index,
    }

    public SortReactions_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.sort_reactions));
    }
}
