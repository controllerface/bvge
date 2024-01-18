package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompleteBoundsMultiBlock_k extends GPUKernel<CompleteBoundsMultiBlock_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        bounds_bank_data(Sizeof.cl_mem),
        sz(Sizeof.cl_mem),
        buffer(Sizeof.cl_mem),
        part(Sizeof.cl_mem),
        n(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CompleteBoundsMultiBlock_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.scan_key_bank.gpu.kernels().get(GPU.Kernel.complete_bounds_multi_block), Args.values());
    }
}
