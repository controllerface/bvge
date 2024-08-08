package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

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

    public PrepareLiquids_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.prepare_liquids));
    }
}
