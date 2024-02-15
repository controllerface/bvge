package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompleteCandidatesMultiBlockOut_k extends GPUKernel<CompleteCandidatesMultiBlockOut_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        input(Sizeof.cl_mem),
        output(Sizeof.cl_mem),
        sz(Sizeof.cl_mem),
        buffer(Sizeof.cl_mem),
        part(Sizeof.cl_mem),
        n(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CompleteCandidatesMultiBlockOut_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.scan_key_candidates.gpu.kernels().get(GPU.Kernel.complete_candidates_multi_block_out), Args.values());
    }
}
