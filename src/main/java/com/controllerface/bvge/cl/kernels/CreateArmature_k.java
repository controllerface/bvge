package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateArmature_k extends GPUKernel<CreateArmature_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        armatures(Sizeof.cl_mem),
        armature_flags(Sizeof.cl_mem),
        hull_tables(Sizeof.cl_mem),
        armature_masses(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_armature(Sizeof.cl_float4),
        new_armature_flags(Sizeof.cl_int4),
        new_hull_table(Sizeof.cl_int2),
        new_armature_mass(Sizeof.cl_float);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CreateArmature_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.gpu_crud.gpu.kernels().get(GPU.Kernel.create_armature), Args.values());
    }
}
