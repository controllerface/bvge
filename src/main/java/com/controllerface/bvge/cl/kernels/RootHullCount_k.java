package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class RootHullCount_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.root_hull_filter;
    private static final GPU.Kernel kernel = GPU.Kernel.root_hull_count;

    public enum Args
    {
        armature_flags,
        counter,
        model_id;
    }

    public RootHullCount_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
