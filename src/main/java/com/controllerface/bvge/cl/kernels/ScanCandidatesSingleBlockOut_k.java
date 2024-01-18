package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class ScanCandidatesSingleBlockOut_k extends GPUKernel<ScanCandidatesSingleBlockOut_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        input(Sizeof.cl_mem),
        output(Sizeof.cl_mem),
        sz(Sizeof.cl_mem),
        buffer(Sizeof.cl_mem),
        n(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public ScanCandidatesSingleBlockOut_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.scan_key_candidates.gpu.kernels().get(GPU.Kernel.scan_candidates_single_block_out), Args.values());
    }
}
