package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CreateHull_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        hull_rotations,
        hull_frictions,
        element_tables,
        hull_flags,
        hull_mesh_ids,
        target,
        new_hull,
        new_rotation,
        new_friction,
        new_table,
        new_flags,
        new_hull_mesh_id;
    }

    public CreateHull_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
