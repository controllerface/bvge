package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class ScanIntMultiBlock_k extends GPUKernel<ScanIntMultiBlock_k.Args>
{
    private static final GPU.Program program = GPU.Program.scan_int_array;
    private static final GPU.Kernel kernel = GPU.Kernel.scan_int_multi_block;

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

    public ScanIntMultiBlock_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
