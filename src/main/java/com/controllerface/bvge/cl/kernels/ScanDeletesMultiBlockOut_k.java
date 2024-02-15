package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class ScanDeletesMultiBlockOut_k extends GPUKernel<ScanDeletesMultiBlockOut_k.Args>
{
    private static final GPU.Program program = GPU.Program.scan_deletes;
    private static final GPU.Kernel kernel = GPU.Kernel.scan_deletes_multi_block_out;

    public enum Args implements GPUKernelArg
    {
        armature_flags(Sizeof.cl_mem),
        hull_tables(Sizeof.cl_mem),
        element_tables(Sizeof.cl_mem),
        hull_flags(Sizeof.cl_mem),
        output(Sizeof.cl_mem),
        output2(Sizeof.cl_mem),
        buffer(Sizeof.cl_mem),
        buffer2(Sizeof.cl_mem),
        part(Sizeof.cl_mem),
        part2(Sizeof.cl_mem),
        n(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public ScanDeletesMultiBlockOut_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
