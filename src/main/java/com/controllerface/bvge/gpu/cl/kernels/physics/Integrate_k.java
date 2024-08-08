package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

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
}
