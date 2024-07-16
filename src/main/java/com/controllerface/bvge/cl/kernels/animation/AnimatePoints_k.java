package com.controllerface.bvge.cl.kernels.animation;

import com.controllerface.bvge.cl.kernels.GPUKernel;

public class AnimatePoints_k extends GPUKernel
{
    public enum Args
    {
        points,
        hull_scales,
        hull_entity_ids,
        hull_flags,
        point_vertex_references,
        point_hull_indices,
        bone_tables,
        vertex_weights,
        entities,
        vertex_references,
        bones,
        max_point,
    }

    public AnimatePoints_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
