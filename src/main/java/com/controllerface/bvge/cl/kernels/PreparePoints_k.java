package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class PreparePoints_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.prepare_points;
    private static final GPU.Kernel kernel = GPU.Kernel.prepare_points;

    public enum Args
    {
        points,
        vertex_vbo,
        offset;
    }

    public PreparePoints_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
