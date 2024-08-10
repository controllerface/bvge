package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class MoveHulls_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        hull_point_tables,
        points,
        max_hull,
    }

    public MoveHulls_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.move_hulls));
    }

    public GPUKernel init()
    {
        return this.buf_arg(Args.hulls, GPU.memory.get_buffer(HULL))
            .buf_arg(Args.hull_point_tables, GPU.memory.get_buffer(HULL_POINT_TABLE))
            .buf_arg(Args.points, GPU.memory.get_buffer(POINT));
    }
}
