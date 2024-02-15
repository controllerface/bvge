package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CountMeshBatches_k extends GPUKernel<CountMeshBatches_k.Args>
{
    private static final GPU.Program program = GPU.Program.mesh_query;
    private static final GPU.Kernel kernel = GPU.Kernel.count_mesh_batches;

    public enum Args implements GPUKernelArg
    {
        mesh_details(Sizeof.cl_mem),
        total(Sizeof.cl_mem),
        max_per_batch(Sizeof.cl_int),
        count(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CountMeshBatches_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
