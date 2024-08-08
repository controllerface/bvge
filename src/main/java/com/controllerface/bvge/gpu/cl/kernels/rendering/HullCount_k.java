package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class HullCount_k extends GPUKernel
{
    public enum Args
    {
        hull_mesh_ids,
        counter,
        mesh_id,
        max_hull,
    }

    public HullCount_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
