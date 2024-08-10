package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.RenderBufferType.*;

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

    public GPUKernel init(CL_Buffer ptr_vbo_transform, CL_Buffer ptr_vbo_color)
    {
        return this.buf_arg(Args.transforms_out, ptr_vbo_transform)
            .buf_arg(Args.colors_out, ptr_vbo_color)
            .buf_arg(Args.hull_positions, GPU.memory.get_buffer(RENDER_HULL))
            .buf_arg(Args.hull_scales, GPU.memory.get_buffer(RENDER_HULL_SCALE))
            .buf_arg(Args.hull_rotations, GPU.memory.get_buffer(RENDER_HULL_ROTATION))
            .buf_arg(Args.hull_point_tables, GPU.memory.get_buffer(RENDER_HULL_POINT_TABLE))
            .buf_arg(Args.hull_uv_offsets, GPU.memory.get_buffer(RENDER_HULL_UV_OFFSET))
            .buf_arg(Args.point_hit_counts, GPU.memory.get_buffer(RENDER_POINT_HIT_COUNT));
    }
}
