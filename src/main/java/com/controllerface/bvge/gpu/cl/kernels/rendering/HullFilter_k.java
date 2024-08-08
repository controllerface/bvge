package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class HullFilter_k extends GPUKernel
{
    public enum Args
    {
        hull_mesh_ids,
        hulls_out,
        counter,
        mesh_id,
        max_hull,
    }

    public HullFilter_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
