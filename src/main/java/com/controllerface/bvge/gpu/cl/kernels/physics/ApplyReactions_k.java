package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class ApplyReactions_k extends GPUKernel
{
    public enum Args
    {
        reactions,
        points,
        anti_gravity,
        point_flags,
        point_hit_counts,
        point_reactions,
        point_offsets,
        point_hull_indices,
        hull_flags,
        max_point,
    }

    public ApplyReactions_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.apply_reactions));
    }
}
