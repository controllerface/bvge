package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.RenderBufferType.RENDER_ENTITY;

public class PrepareEntities_k extends GPUKernel
{
    public enum Args
    {
        points,
        vertex_vbo,
        offset,
        max_entity,
    }

    public PrepareEntities_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.prepare_entities));
    }

    public GPUKernel init(CL_Buffer ptr_vbo_vertex)
    {
        return this.buf_arg(PrepareEntities_k.Args.vertex_vbo, ptr_vbo_vertex)
            .buf_arg(PrepareEntities_k.Args.points, GPU.memory.get_buffer(RENDER_ENTITY));
    }
}
