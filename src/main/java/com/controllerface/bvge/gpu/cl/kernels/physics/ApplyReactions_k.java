package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

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

    public ApplyReactions_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
