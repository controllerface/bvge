package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class PrepareBounds_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.prepare_bounds;
    private static final GPU.Kernel kernel = GPU.Kernel.prepare_bounds;

    public enum Args
    {
        bounds,
        vbo,
        offset;
    }

    public PrepareBounds_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
