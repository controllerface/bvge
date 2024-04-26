package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class PrepareLiquids_k extends GPUKernel
{
    public enum Args
    {
        hull_positions,
        hull_scales,
        hull_rotations,
        hull_point_tables,
        point_hit_counts,
        indices,
        transforms_out,
        colors_out,
        offset;
    }

    public PrepareLiquids_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
