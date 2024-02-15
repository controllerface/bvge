package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompactPoints_k extends GPUKernel<CompactPoints_k.Args>
{
    private static final GPU.Program program = GPU.Program.scan_deletes;
    private static final GPU.Kernel kernel = GPU.Kernel.compact_points;

    public enum Args implements GPUKernelArg
    {
        point_shift(Sizeof.cl_mem),
        points(Sizeof.cl_mem),
        anti_gravity(Sizeof.cl_mem),
        vertex_tables(Sizeof.cl_mem),
        bone_tables(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CompactPoints_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
