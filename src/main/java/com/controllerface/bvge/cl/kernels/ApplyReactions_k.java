package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class ApplyReactions_k extends GPUKernel<ApplyReactions_k.Args>
{
    private static final GPU.Program program = GPU.Program.sat_collide;
    private static final GPU.Kernel kernel = GPU.Kernel.apply_reactions;

    public enum Args implements GPUKernelArg
    {
        reactions(Sizeof.cl_float2),
        points(Sizeof.cl_float4),
        anti_gravity(Sizeof.cl_float),
        point_reactions(Sizeof.cl_int),
        point_offsets(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public ApplyReactions_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
