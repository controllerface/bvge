package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class PrepareLiquids_k extends GPUKernel
{
    public enum Args
    {
        hull_positions,
        hull_scales,
        hull_rotations,
        hull_point_tables,
        hull_uv_offsets,
        point_hit_counts,
        indices,
        transforms_out,
        colors_out,
        offset,
        max_hull,
    }

    public PrepareLiquids_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
