package com.controllerface.bvge.gpu.cl.kernels.egress;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class EgressEdges_k extends GPUKernel
{
    public enum Args
    {
        edges_in,
        edge_lengths_in,
        edge_flags_in,
        edge_pins_in,
        edges_out,
        edge_lengths_out,
        edge_flags_out,
        edge_pins_out,
        new_edges,
        max_edge,
    }

    public EgressEdges_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.egress_edges));
    }
}
