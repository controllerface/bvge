package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class MoveEntities_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        entities,
        entity_flags,
        entity_motion_states,
        entity_hull_tables,
        hull_point_tables,
        hull_integrity,
        hull_flags,
        point_flags,
        point_hit_counts,
        points,
        dt,
        max_entity,
    }

    public MoveEntities_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.move_entities));
    }
}
