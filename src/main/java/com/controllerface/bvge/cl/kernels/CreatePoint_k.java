package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CreatePoint_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.gpu_crud;
    private static final GPU.Kernel kernel = GPU.Kernel.create_point;

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

    public CreatePoint_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
