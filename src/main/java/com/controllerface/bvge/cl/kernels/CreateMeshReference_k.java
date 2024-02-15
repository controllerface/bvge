package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateMeshReference_k extends GPUKernel<CreateMeshReference_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        mesh_ref_tables(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_mesh_ref_table(Sizeof.cl_float4);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CreateMeshReference_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.gpu_crud.gpu.kernels().get(GPU.Kernel.create_mesh_reference), Args.values());
    }
}
