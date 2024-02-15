package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class GenerateKeys_k extends GPUKernel<GenerateKeys_k.Args>
{
    private static final GPU.Program program = GPU.Program.generate_keys;
    private static final GPU.Kernel kernel = GPU.Kernel.generate_keys;

    public enum Args implements GPUKernelArg
    {
        bounds_index_data(Sizeof.cl_mem),
        bounds_bank_data(Sizeof.cl_mem),
        key_bank(Sizeof.cl_mem),
        key_counts(Sizeof.cl_mem),
        x_subdivisions(Sizeof.cl_int),
        key_bank_length(Sizeof.cl_int),
        key_count_length(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public GenerateKeys_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
