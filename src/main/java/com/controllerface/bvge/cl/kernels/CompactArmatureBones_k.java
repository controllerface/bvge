package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompactArmatureBones_k extends GPUKernel<CompactArmatureBones_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        armature_bone_shift(Sizeof.cl_mem),
        armature_bones(Sizeof.cl_mem),
        armature_bone_tables(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CompactArmatureBones_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.scan_deletes.gpu.kernels().get(GPU.Kernel.compact_armature_bones), Args.values());
    }
}
