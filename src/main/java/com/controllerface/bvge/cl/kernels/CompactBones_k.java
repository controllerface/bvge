package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompactBones_k extends GPUKernel<CompactBones_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        bone_shift(Sizeof.cl_mem),
        bone_instances(Sizeof.cl_mem),
        bone_index_tables(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CompactBones_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.scan_deletes.gpu.kernels().get(GPU.Kernel.compact_bones), Args.values());
    }
}
