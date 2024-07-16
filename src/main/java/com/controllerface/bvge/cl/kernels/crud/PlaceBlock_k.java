package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.kernels.GPUKernel;

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

    public PlaceBlock_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
