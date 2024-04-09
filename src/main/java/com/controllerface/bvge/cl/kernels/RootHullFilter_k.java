package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class RootHullFilter_k extends GPUKernel
{
    public enum Args
    {
        armature_root_hulls,
        armature_model_indices,
        hulls_out,
        counter,
        model_id;
    }

    public RootHullFilter_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
