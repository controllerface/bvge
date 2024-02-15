package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class RootHullCount_k extends GPUKernel<RootHullCount_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        armature_flags(Sizeof.cl_mem),
        counter(Sizeof.cl_mem),
        model_id(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public RootHullCount_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.root_hull_filter.gpu.kernels().get(GPU.Kernel.root_hull_count), Args.values());
    }
}
