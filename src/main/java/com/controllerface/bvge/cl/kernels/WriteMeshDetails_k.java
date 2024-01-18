package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class WriteMeshDetails_k extends GPUKernel<WriteMeshDetails_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        hull_mesh_ids(Sizeof.cl_mem),
        mesh_references(Sizeof.cl_mem),
        counters(Sizeof.cl_mem),
        query(Sizeof.cl_mem),
        offsets(Sizeof.cl_mem),
        mesh_details(Sizeof.cl_mem),
        count(Sizeof.cl_int),;

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public WriteMeshDetails_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.write_mesh_details), Args.values());
    }
}
