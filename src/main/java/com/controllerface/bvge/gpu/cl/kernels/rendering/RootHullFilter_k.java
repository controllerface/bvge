package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class RootHullFilter_k extends GPUKernel
{
    public enum Args
    {
        entity_root_hulls,
        entity_model_indices,
        hulls_out,
        counter,
        model_id,
        max_entity,
    }

    public RootHullFilter_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.root_hull_filter));
    }
}
