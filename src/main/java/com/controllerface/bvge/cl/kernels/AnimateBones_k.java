package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class AnimateBones_k extends GPUKernel<AnimateBones_k.Args>
{
    private static final GPU.Program program = GPU.Program.animate_hulls;
    private static final GPU.Kernel kernel = GPU.Kernel.animate_bones;

    public enum Args implements GPUKernelArg
    {
        bones(Sizeof.cl_mem),
        bone_references(Sizeof.cl_mem),
        armature_bones(Sizeof.cl_mem),
        bone_index_tables(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public AnimateBones_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
