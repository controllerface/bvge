package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class FinalizeCandidates_k extends GPUKernel
{
    public enum Args
    {
        input_candidates,
        match_offsets,
        matches,
        used,
        counter,
        final_candidates,
        max_index,
    }

    public FinalizeCandidates_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
