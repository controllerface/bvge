package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.types.RenderBufferType;

public class PrepareBounds_k extends GPUKernel
{
    public enum Args
    {
        bounds,
        vbo,
        offset,
        max_bound,
    }

    public PrepareBounds_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.prepare_bounds));
    }

    public GPUKernel init(CL_Buffer ptr_vbo_position)
    {
        return this.buf_arg(PrepareBounds_k.Args.vbo, ptr_vbo_position)
            .buf_arg(PrepareBounds_k.Args.bounds, GPU.memory.get_buffer(RenderBufferType.RENDER_HULL_AABB));
    }
}
