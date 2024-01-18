package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CountMeshInstances_k extends GPUKernel<CountMeshInstances_k.Args>
{
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

    public CountMeshInstances_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.count_mesh_instances), Args.values());
    }

}
