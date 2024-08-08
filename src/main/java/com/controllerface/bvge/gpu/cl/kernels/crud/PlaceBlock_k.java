package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class PlaceBlock_k extends GPUKernel
{
    public enum Args
    {
        entities,
        entity_hull_tables,
        hulls,
        hull_point_tables,
        hull_rotations,
        points,
        src,
        dest,
    }

    public PlaceBlock_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
