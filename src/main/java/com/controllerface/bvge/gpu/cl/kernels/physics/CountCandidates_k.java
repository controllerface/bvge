package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class CountCandidates_k extends GPUKernel
{
    public enum Args
    {
        bounds_bank_data,
        in_bounds,
        key_bank,
        key_counts,
        candidates,
        x_subdivisions,
        key_count_length,
        max_index,
    }

    public CountCandidates_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.count_candidates));
    }
}
