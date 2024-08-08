package com.controllerface.bvge.gpu.cl.kernels.compact;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class ScanDeletesSingleBlockOut_k extends GPUKernel
{
    public enum Args
    {
        entity_flags,
        hull_tables,
        bone_tables,
        point_tables,
        edge_tables,
        hull_bone_tables,
        output,
        output2,
        sz,
        buffer,
        buffer2,
        n;
    }

    public ScanDeletesSingleBlockOut_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.scan_deletes_single_block_out));
    }
}
