package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class ResolveConstraints_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.resolve_constraints;
    private static final GPU.Kernel kernel = GPU.Kernel.resolve_constraints;

    public enum Args
    {
        element_table,
        bounds_bank_dat,
        point,
        edges,
        process_all;
    }

    public ResolveConstraints_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
