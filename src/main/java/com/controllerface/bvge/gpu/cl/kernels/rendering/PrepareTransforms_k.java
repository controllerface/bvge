package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.RenderBufferType.*;

public class PrepareTransforms_k extends GPUKernel
{
    public enum Args
    {
        hull_positions,
        hull_scales,
        hull_rotations,
        indices,
        transforms_out,
        offset,
        max_hull,
    }

    public PrepareTransforms_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.prepare_transforms));
    }

    public GPUKernel init(CL_Buffer ptr_vbo_transforms)
    {
        return this.buf_arg(Args.transforms_out, ptr_vbo_transforms)
            .buf_arg(Args.hull_positions, GPU.memory.get_buffer(RENDER_HULL))
            .buf_arg(Args.hull_scales, GPU.memory.get_buffer(RENDER_HULL_SCALE))
            .buf_arg(Args.hull_rotations, GPU.memory.get_buffer(RENDER_HULL_ROTATION));
    }
}
