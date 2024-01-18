package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class LocateOutOfBounds_k extends GPUKernel<LocateOutOfBounds_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        hull_tables(Sizeof.cl_mem),
        hull_flags(Sizeof.cl_mem),
        armature_flags(Sizeof.cl_mem),
        counter(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public LocateOutOfBounds_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.locate_out_of_bounds), Args.values());
    }
}
