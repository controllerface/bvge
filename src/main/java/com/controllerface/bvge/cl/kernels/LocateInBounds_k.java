package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class LocateInBounds_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.locate_in_bounds;
    private static final GPU.Kernel kernel = GPU.Kernel.locate_in_bounds;

    public enum Args
    {
        bounds_bank_data,
        in_bounds,
        counter;
    }

    public LocateInBounds_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
