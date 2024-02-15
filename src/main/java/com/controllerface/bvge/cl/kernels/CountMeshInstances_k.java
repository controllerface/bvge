package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CountMeshInstances_k extends GPUKernel<CountMeshInstances_k.Args>
{
    private static final GPU.Program program = GPU.Program.mesh_query;
    private static final GPU.Kernel kernel = GPU.Kernel.count_mesh_instances;

    public enum Args implements GPUKernelArg
    {
        hull_mesh_ids(Sizeof.cl_mem),
        counters(Sizeof.cl_mem),
        query(Sizeof.cl_mem),
        total(Sizeof.cl_mem),
        count(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CountMeshInstances_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
