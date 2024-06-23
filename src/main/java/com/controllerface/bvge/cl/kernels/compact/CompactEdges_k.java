package com.controllerface.bvge.cl.kernels.compact;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.crud.KernelArg;

public class CompactEdges_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.compact_k_src(Kernel.compact_edges, Args.class);

    public enum Args implements KernelArg
    {
        edge_shift   (Type.buffer_int),
        edges        (Type.buffer_int2),
        edge_lengths (Type.buffer_float),
        edge_flags   (Type.buffer_int),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CompactEdges_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
