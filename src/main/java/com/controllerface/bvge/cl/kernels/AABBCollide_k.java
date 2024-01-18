package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class AABBCollide_k extends GPUKernel<AABBCollide_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        bounds(Sizeof.cl_mem),
        bounds_bank_data(Sizeof.cl_mem),
        hull_flags(Sizeof.cl_mem),
        candidates(Sizeof.cl_mem),
        match_offsets(Sizeof.cl_mem),
        key_map(Sizeof.cl_mem),
        key_bank(Sizeof.cl_mem),
        key_counts(Sizeof.cl_mem),
        key_offsets(Sizeof.cl_mem),
        matches(Sizeof.cl_mem),
        used(Sizeof.cl_mem),
        counter(Sizeof.cl_mem),
        x_subdivisions(Sizeof.cl_int),
        key_count_length(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return this.size; }
    }

    public AABBCollide_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.aabb_collide), Args.values());
    }
}
