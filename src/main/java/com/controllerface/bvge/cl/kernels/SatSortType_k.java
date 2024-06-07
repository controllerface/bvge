package com.controllerface.bvge.cl.kernels;

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

    public SatSortType_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
