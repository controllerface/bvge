package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CompleteDeletesMultiBlockOut_k extends GPUKernel
{
    public enum Args
    {
        armature_flags,
        hull_tables,
        bone_tables,
        element_tables,
        hull_bone_tables,
        output1,
        output2,
        sz,
        buffer1,
        buffer2,
        part1,
        part2,
        n;
    }

    public CompleteDeletesMultiBlockOut_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
