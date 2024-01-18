package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class LocateInBounds_k extends GPUKernel<LocateInBounds_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        bounds_bank_data(Sizeof.cl_mem),
        in_bounds(Sizeof.cl_mem),
        counter(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public LocateInBounds_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.locate_in_bounds), Args.values());
    }
}
