package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class ApplyReactions_k extends GPUKernel
{
    public enum Arg
    {
        reactions(Sizeof.cl_float2),
        points(Sizeof.cl_float4),
        anti_gravity(Sizeof.cl_float),
        point_reactions(Sizeof.cl_int),
        point_offsets(Sizeof.cl_int);

        public final long size;

        Arg(long size)
        {
            this.size = size;
        }
    }

    public ApplyReactions_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.apply_reactions), Arg.values().length);
        for (var arg : Arg.values())
        {
            def_arg(arg.ordinal(), arg.size);
        }
    }
}
