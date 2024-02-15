package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompactEdges_k extends GPUKernel<CompactEdges_k.Args>
{
    private static final GPU.Program program = GPU.Program.scan_deletes;
    private static final GPU.Kernel kernel = GPU.Kernel.compact_edges;

    public enum Args implements GPUKernelArg
    {
        edge_shift(Sizeof.cl_mem),
        edges(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CompactEdges_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
