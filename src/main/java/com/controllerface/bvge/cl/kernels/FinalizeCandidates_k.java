package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class FinalizeCandidates_k extends GPUKernel<FinalizeCandidates_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        input_candidates(Sizeof.cl_mem),
        match_offsets(Sizeof.cl_mem),
        matches(Sizeof.cl_mem),
        used(Sizeof.cl_mem),
        counter(Sizeof.cl_mem),
        final_candidates(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public FinalizeCandidates_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.locate_in_bounds.gpu.kernels().get(GPU.Kernel.finalize_candidates), Args.values());
    }
}
