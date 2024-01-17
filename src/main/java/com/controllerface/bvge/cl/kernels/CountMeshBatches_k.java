package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CountMeshBatches_k extends GPUKernel
{
    public enum Arg
    {
        mesh_details(Sizeof.cl_mem),
        total(Sizeof.cl_mem),
        count(Sizeof.cl_int);

        public final long size;

        Arg(long size)
        {
            this.size = size;
        }
    }

    public CountMeshBatches_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.count_mesh_batches), Arg.values().length);
        for (var arg : Arg.values())
        {
            def_arg(arg.ordinal(), arg.size);
        }
    }
}
