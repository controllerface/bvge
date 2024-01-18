package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class PrepareEdges_k extends GPUKernel<PrepareEdges_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        points(Sizeof.cl_mem),
        edges(Sizeof.cl_mem),
        vertex_vbo(Sizeof.cl_mem),
        flag_vbo(Sizeof.cl_mem),
        offset(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public PrepareEdges_k(cl_command_queue command_queue)
    {
        super(command_queue,  GPU.Program.prepare_edges.gpu.kernels().get(GPU.Kernel.prepare_edges), Args.values());
    }
}
