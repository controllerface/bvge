package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class LocateOutOfBounds_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.scan_deletes;
    private static final GPU.Kernel kernel = GPU.Kernel.locate_out_of_bounds;

    public enum Args
    {
        hull_tables,
        hull_flags,
        armature_flags,
        counter;
    }

    public LocateOutOfBounds_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
