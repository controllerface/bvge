package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CountCandidates_k extends GPUKernel<CountCandidates_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        bounds_bank_data(Sizeof.cl_mem),
        in_bounds(Sizeof.cl_mem),
        key_bank(Sizeof.cl_mem),
        key_counts(Sizeof.cl_mem),
        candidates(Sizeof.cl_mem),
        x_subdivisions(Sizeof.cl_int),
        key_count_length(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CountCandidates_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.locate_in_bounds.gpu.kernels().get(GPU.Kernel.count_candidates), Args.values());
    }
}
