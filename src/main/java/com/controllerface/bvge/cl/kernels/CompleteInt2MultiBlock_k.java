package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompleteInt2MultiBlock_k extends GPUKernel<CompleteInt2MultiBlock_k.Args>
{
    private static final GPU.Program program = GPU.Program.scan_int2_array;
    private static final GPU.Kernel kernel = GPU.Kernel.complete_int2_multi_block;

    public enum Args implements GPUKernelArg
    {
        data(Sizeof.cl_mem),
        buffer(Sizeof.cl_mem),
        part(Sizeof.cl_mem),
        n(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CompleteInt2MultiBlock_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
