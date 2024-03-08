package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class ScanDeletesMultiBlockOut_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.scan_deletes;
    private static final GPU.Kernel kernel = GPU.Kernel.scan_deletes_multi_block_out;

    public enum Args
    {
        armature_flags,
        hull_tables,
        element_tables,
        hull_flags,
        output,
        output2,
        buffer,
        buffer2,
        part,
        part2,
        n;
    }

    public ScanDeletesMultiBlockOut_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
