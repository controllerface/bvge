package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class ScanCandidatesMultiBlockOut_k extends GPUKernel<ScanCandidatesMultiBlockOut_k.Args>
{
    private static final GPU.Program program = GPU.Program.scan_key_candidates;
    private static final GPU.Kernel kernel = GPU.Kernel.scan_candidates_multi_block_out;

    public enum Args implements GPUKernelArg
    {
        input(Sizeof.cl_mem),
        output(Sizeof.cl_mem),
        buffer(Sizeof.cl_mem),
        part(Sizeof.cl_mem),
        n(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public ScanCandidatesMultiBlockOut_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
