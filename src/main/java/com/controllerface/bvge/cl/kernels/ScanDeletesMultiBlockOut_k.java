package com.controllerface.bvge.cl.kernels;

public class ScanDeletesMultiBlockOut_k extends GPUKernel
{
    public enum Args
    {
        entity_flags,
        hull_tables,
        bone_tables,
        point_tables,
        edge_tables,
        hull_bone_tables,
        output1,
        output2,
        buffer1,
        buffer2,
        part1,
        part2,
        n;
    }

    public ScanDeletesMultiBlockOut_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
