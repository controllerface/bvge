package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class Integrate_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        armatures,
        armature_flags,
        element_tables,
        armature_accel,
        hull_rotations,
        points,
        bounds,
        bounds_index_data,
        bounds_bank_data,
        hull_flags,
        anti_gravity,
        args;
    }

    public Integrate_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
