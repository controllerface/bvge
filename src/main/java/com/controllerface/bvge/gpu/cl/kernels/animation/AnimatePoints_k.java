package com.controllerface.bvge.gpu.cl.kernels.animation;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

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

    public AnimatePoints_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
