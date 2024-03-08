package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class PrepareTransforms_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.prepare_transforms;
    private static final GPU.Kernel kernel = GPU.Kernel.prepare_transforms;

    public enum Args
    {
        transforms,
        hull_rotations,
        indices,
        transforms_out,
        offset;
    }

    public PrepareTransforms_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
