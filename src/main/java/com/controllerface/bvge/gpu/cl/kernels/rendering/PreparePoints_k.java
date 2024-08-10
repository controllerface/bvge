package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.RenderBufferType.RENDER_POINT;
import static com.controllerface.bvge.memory.types.RenderBufferType.RENDER_POINT_ANTI_GRAV;

public class PreparePoints_k extends GPUKernel
{
    public enum Args
    {
        points,
        anti_gravity,
        vertex_vbo,
        color_vbo,
        offset,
        max_point,
    }

    public PreparePoints_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.prepare_points));
    }

    public GPUKernel init(CL_Buffer cl_vertex_vbo, CL_Buffer cl_color_vbo)
    {
        return this.buf_arg(PreparePoints_k.Args.vertex_vbo, cl_vertex_vbo)
            .buf_arg(PreparePoints_k.Args.color_vbo, cl_color_vbo)
            .buf_arg(PreparePoints_k.Args.anti_gravity, GPU.memory.get_buffer(RENDER_POINT_ANTI_GRAV))
            .buf_arg(PreparePoints_k.Args.points, GPU.memory.get_buffer(RENDER_POINT));
    }
}
