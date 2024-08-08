package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class SatCollide_k extends GPUKernel
{
    public enum Args
    {
        candidates,
        entity_model_transforms,
        entity_flags,
        hulls,
        hull_scales,
        hull_frictions,
        hull_restitutions,
        hull_integrity,
        hull_point_tables,
        hull_edge_tables,
        hull_entity_ids,
        hull_flags,
        point_flags,
        points,
        edges,
        edge_flags,
        reactions,
        reaction_index,
        point_reactions,
        masses,
        counter,
        dt,
        max_index,
    }

    public SatCollide_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.sat_collide));
    }
}
