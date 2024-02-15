package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompactBones_k extends GPUKernel<CompactBones_k.Args>
{
    private static final GPU.Program program = GPU.Program.scan_deletes;
    private static final GPU.Kernel kernel = GPU.Kernel.compact_bones;

    public enum Args implements GPUKernelArg
    {
        bone_shift(Sizeof.cl_mem),
        bone_instances(Sizeof.cl_mem),
        bone_index_tables(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CompactBones_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
