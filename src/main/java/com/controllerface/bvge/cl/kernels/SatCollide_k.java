package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class SatCollide_k extends GPUKernel<SatCollide_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        candidates(Sizeof.cl_mem),
        hulls(Sizeof.cl_mem),
        element_tables(Sizeof.cl_mem),
        hull_flags(Sizeof.cl_mem),
        vertex_tables(Sizeof.cl_mem),
        points(Sizeof.cl_mem),
        edges(Sizeof.cl_mem),
        reactions(Sizeof.cl_mem),
        reaction_index(Sizeof.cl_mem),
        point_reactions(Sizeof.cl_mem),
        masses(Sizeof.cl_mem),
        counter(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public SatCollide_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.sat_collide.gpu.kernels().get(GPU.Kernel.sat_collide), Args.values());
    }
}
