package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CreatePoint_k extends GPUKernel
{
    public enum Args
    {
        points,
        vertex_tables,
        bone_tables,
        target,
        new_point,
        new_vertex_table,
        new_bone_table;
    }

    public CreatePoint_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
