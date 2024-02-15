package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class RootHullFilter_k extends GPUKernel<RootHullFilter_k.Args>
{
    private static final GPU.Program program = GPU.Program.root_hull_filter;
    private static final GPU.Kernel kernel = GPU.Kernel.root_hull_filter;

    public enum Args implements GPUKernelArg
    {
        armature_flags(Sizeof.cl_mem),
        hulls_out(Sizeof.cl_mem),
        counter(Sizeof.cl_mem),
        model_id(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public RootHullFilter_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
