package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class SatSortType_k extends GPUKernel
{
    public enum Args
    {
        candidates,
        hull_flags,
        sat_candidates_p,
        sat_candidates_c,
        sat_candidates_b,
        sat_candidates_pc,
        counter,
    }

    public SatSortType_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
