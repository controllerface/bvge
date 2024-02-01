package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class AnimateArmatures_k extends GPUKernel<AnimateArmatures_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        armature_bones(Sizeof.cl_mem),
        bone_bind_poses(Sizeof.cl_mem),
        bone_bind_tables(Sizeof.cl_mem),
        hull_tables(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public AnimateArmatures_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.animate_hulls.gpu.kernels().get(GPU.Kernel.animate_points), Args.values());
    }
}
