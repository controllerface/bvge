package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateEdge_k extends GPUKernel<CreateEdge_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        edges(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_edge(Sizeof.cl_float4);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CreateEdge_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.create_edge), Args.values());
    }
}
