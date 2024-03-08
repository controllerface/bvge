package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class ScanDeletesSingleBlockOut_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.scan_deletes;
    private static final GPU.Kernel kernel = GPU.Kernel.scan_deletes_single_block_out;

    public enum Args
    {
        armature_flags,
        hull_tables,
        element_tables,
        hull_flags,
        output,
        output2,
        sz,
        buffer,
        buffer2,
        n;
    }

    public ScanDeletesSingleBlockOut_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
