package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompleteIntMultiBlockOut_k extends GPUKernel<CompleteIntMultiBlockOut_k.Args>
{
    private static final GPU.Program program = GPU.Program.scan_int_array_out;
    private static final GPU.Kernel kernel = GPU.Kernel.complete_int_multi_block_out;

    public enum Args implements GPUKernelArg
    {
        output(Sizeof.cl_mem),
        buffer(Sizeof.cl_mem),
        part(Sizeof.cl_mem),
        n(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CompleteIntMultiBlockOut_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
