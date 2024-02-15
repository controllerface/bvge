package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class ResolveConstraints_k extends GPUKernel<ResolveConstraints_k.Args>
{
    private static final GPU.Program program = GPU.Program.resolve_constraints;
    private static final GPU.Kernel kernel = GPU.Kernel.resolve_constraints;

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

    public ResolveConstraints_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
