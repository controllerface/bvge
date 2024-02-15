package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class BuildKeyMap_k extends GPUKernel<BuildKeyMap_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        bounds_index_data(Sizeof.cl_mem),
        bounds_bank_data(Sizeof.cl_mem),
        key_map(Sizeof.cl_mem),
        key_offsets(Sizeof.cl_mem),
        key_counts(Sizeof.cl_mem),
        x_subdivisions(Sizeof.cl_int),
        key_count_length(Sizeof.cl_int),
        ;

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public BuildKeyMap_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.build_key_map.gpu.kernels().get(GPU.Kernel.build_key_map), Args.values());
    }
}
