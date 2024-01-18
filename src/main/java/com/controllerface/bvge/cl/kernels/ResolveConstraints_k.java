package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class ResolveConstraints_k extends GPUKernel<ResolveConstraints_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        element_table(Sizeof.cl_mem),
        bounds_bank_dat(Sizeof.cl_mem),
        point(Sizeof.cl_mem),
        edges(Sizeof.cl_mem),
        process_all(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public ResolveConstraints_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.resolve_constraints), Args.values());
    }
}
