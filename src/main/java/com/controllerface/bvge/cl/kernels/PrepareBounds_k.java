package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class PrepareBounds_k extends GPUKernel<PrepareBounds_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        bounds(Sizeof.cl_mem),
        vbo(Sizeof.cl_mem),
        offset(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public PrepareBounds_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.prepare_bounds.gpu.kernels().get(GPU.Kernel.prepare_bounds), Args.values());
    }
}
