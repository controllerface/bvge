package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class MoveArmatures_k extends GPUKernel<MoveArmatures_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        hulls(Sizeof.cl_mem),
        armatures(Sizeof.cl_mem),
        hull_tables(Sizeof.cl_mem),
        element_tables(Sizeof.cl_mem),
        hull_flags(Sizeof.cl_mem),
        points(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public MoveArmatures_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.sat_collide.gpu.kernels().get(GPU.Kernel.move_armatures), Args.values());
    }
}
