package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class ScanInt4SingleBlock_k extends GPUKernel<ScanInt4SingleBlock_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        data(Sizeof.cl_mem),
        buffer(Sizeof.cl_mem),
        n(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public ScanInt4SingleBlock_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.scan_int4_array.gpu.kernels().get(GPU.Kernel.scan_int4_single_block), Args.values());
    }
}
