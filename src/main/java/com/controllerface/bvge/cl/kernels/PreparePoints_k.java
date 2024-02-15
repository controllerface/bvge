package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class PreparePoints_k extends GPUKernel<PreparePoints_k.Args>
{
    private static final GPU.Program program = GPU.Program.prepare_points;
    private static final GPU.Kernel kernel = GPU.Kernel.prepare_points;

    public enum Args implements GPUKernelArg
    {
        points(Sizeof.cl_mem),
        vertex_vbo(Sizeof.cl_mem),
        offset(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public PreparePoints_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
