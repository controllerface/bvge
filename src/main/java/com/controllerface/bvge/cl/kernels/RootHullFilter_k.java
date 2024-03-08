package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class RootHullFilter_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.root_hull_filter;
    private static final GPU.Kernel kernel = GPU.Kernel.root_hull_filter;

    public enum Args
    {
        armature_flags,
        hulls_out,
        counter,
        model_id;
    }

    public RootHullFilter_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
