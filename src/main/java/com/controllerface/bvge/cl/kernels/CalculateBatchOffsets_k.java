package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CalculateBatchOffsets_k extends GPUKernel<CalculateBatchOffsets_k.Args>
{
    private static final GPU.Program program = GPU.Program.mesh_query;
    private static final GPU.Kernel kernel = GPU.Kernel.calculate_batch_offsets;

    public enum Args implements GPUKernelArg
    {
        mesh_offsets(Sizeof.cl_mem),
        mesh_details(Sizeof.cl_mem),
        count(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CalculateBatchOffsets_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
