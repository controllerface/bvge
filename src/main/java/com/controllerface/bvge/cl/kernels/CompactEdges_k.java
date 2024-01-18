package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompactEdges_k extends GPUKernel<CompactEdges_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        edge_shift(Sizeof.cl_mem),
        edges(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CompactEdges_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.scan_deletes.gpu.kernels().get(GPU.Kernel.compact_edges), Args.values());
    }
}
