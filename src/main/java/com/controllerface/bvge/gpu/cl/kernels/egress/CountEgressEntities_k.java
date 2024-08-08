package com.controllerface.bvge.gpu.cl.kernels.egress;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class CountEgressEntities_k extends GPUKernel
{
    public enum Args
    {
        entity_flags,
        entity_hull_tables,
        entity_bone_tables,
        hull_flags,
        hull_point_tables,
        hull_edge_tables,
        hull_bone_tables,
        counters,
        max_entity,
    }

    public CountEgressEntities_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.count_egress_entities));
    }
}
