package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class ScanDeletesSingleBlockOut_k extends GPUKernel
{
    public enum Args
    {
        armature_flags,
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

    public ScanDeletesSingleBlockOut_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
