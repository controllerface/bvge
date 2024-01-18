package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateHull_k extends GPUKernel<CreateHull_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        hulls(Sizeof.cl_mem),
        hull_rotations(Sizeof.cl_mem),
        element_tables(Sizeof.cl_mem),
        hull_flags(Sizeof.cl_mem),
        hull_mesh_ids(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_hull(Sizeof.cl_float4),
        new_rotation(Sizeof.cl_float2),
        new_table(Sizeof.cl_int4),
        new_flags(Sizeof.cl_int4),
        new_hull_mesh_id(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CreateHull_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.gpu_crud.gpu.kernels().get(GPU.Kernel.create_hull), Args.values());
    }
}
