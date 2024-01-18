package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class AnimateHulls_k extends GPUKernel<AnimateHulls_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        points(Sizeof.cl_mem),
        hulls(Sizeof.cl_mem),
        hull_flags(Sizeof.cl_mem),
        vertex_tables(Sizeof.cl_mem),
        bone_tables(Sizeof.cl_mem),
        armatures(Sizeof.cl_mem),
        vertex_references(Sizeof.cl_mem),
        bones(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public AnimateHulls_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.animate_hulls.gpu.kernels().get(GPU.Kernel.animate_hulls), Args.values());
    }
}
