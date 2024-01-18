package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class PrepareBones_k extends GPUKernel<PrepareBones_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        bones(Sizeof.cl_mem),
        bone_references(Sizeof.cl_mem),
        bone_index(Sizeof.cl_mem),
        hulls(Sizeof.cl_mem),
        armatures(Sizeof.cl_mem),
        hull_flags(Sizeof.cl_mem),
        vbo(Sizeof.cl_mem),
        offset(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public PrepareBones_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.prepare_bones.gpu.kernels().get(GPU.Kernel.prepare_bones), Args.values());
    }
}
