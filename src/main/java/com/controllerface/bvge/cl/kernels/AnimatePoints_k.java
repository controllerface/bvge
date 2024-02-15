package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class AnimatePoints_k extends GPUKernel<AnimatePoints_k.Args>
{
    private static final GPU.Program program = GPU.Program.animate_hulls;
    private static final GPU.Kernel kernel = GPU.Kernel.animate_points;

    public enum Args implements GPUKernelArg
    {
        points(Sizeof.cl_mem),
        hulls(Sizeof.cl_mem),
        hull_flags(Sizeof.cl_mem),
        vertex_tables(Sizeof.cl_mem),
        bone_tables(Sizeof.cl_mem),
        vertex_weights(Sizeof.cl_mem),
        armatures(Sizeof.cl_mem),
        vertex_references(Sizeof.cl_mem),
        bones(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public AnimatePoints_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
