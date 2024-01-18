package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class Integrate_k extends GPUKernel<Integrate_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        hulls(Sizeof.cl_mem),
        armatures(Sizeof.cl_mem),
        armature_flags(Sizeof.cl_mem),
        element_tables(Sizeof.cl_mem),
        armature_accel(Sizeof.cl_mem),
        hull_rotations(Sizeof.cl_mem),
        points(Sizeof.cl_mem),
        bounds(Sizeof.cl_mem),
        bounds_index_data(Sizeof.cl_mem),
        bounds_bank_data(Sizeof.cl_mem),
        hull_flags(Sizeof.cl_mem),
        anti_gravity(Sizeof.cl_mem),
        args(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public Integrate_k(cl_command_queue command_queue, GPUProgram program)
    {
        super(command_queue, program.kernels().get(GPU.Kernel.integrate), Args.values());
    }
}
