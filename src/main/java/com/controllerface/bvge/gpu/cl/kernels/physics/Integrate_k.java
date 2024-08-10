package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class Integrate_k extends GPUKernel
{
    public enum Args
    {
        hull_point_tables,
        entity_accel,
        points,
        point_hit_counts,
        point_flags,
        hull_flags,
        hull_entity_ids,
        anti_gravity,
        args,
        max_hull,
    }

    public Integrate_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.integrate));
    }

    public GPUKernel init()
    {
        return this.buf_arg(Args.hull_point_tables, GPU.memory.get_buffer(HULL_POINT_TABLE))
            .buf_arg(Args.entity_accel, GPU.memory.get_buffer(ENTITY_ACCEL))
            .buf_arg(Args.points, GPU.memory.get_buffer(POINT))
            .buf_arg(Args.point_hit_counts, GPU.memory.get_buffer(POINT_HIT_COUNT))
            .buf_arg(Args.point_flags, GPU.memory.get_buffer(POINT_FLAG))
            .buf_arg(Args.hull_flags, GPU.memory.get_buffer(HULL_FLAG))
            .buf_arg(Args.hull_entity_ids, GPU.memory.get_buffer(HULL_ENTITY_ID))
            .buf_arg(Args.anti_gravity, GPU.memory.get_buffer(POINT_ANTI_GRAV));
    }
}
