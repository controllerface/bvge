package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class ResolveConstraints_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        entities,
        hull_edge_tables,
        bounds_bank_data,
        points,
        edges,
        edge_lengths,
        edge_flags,
        edge_pins,
        process_all,
        max_hull,
    }

    public ResolveConstraints_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.resolve_constraints));
    }
}
