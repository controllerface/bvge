package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class AABBCollide_k extends GPUKernel
{
    public enum Args
    {
        bounds,
        bounds_bank_data,
        hull_entity_ids,
        hull_flags,
        candidates,
        match_offsets,
        key_map,
        key_bank,
        key_counts,
        key_offsets,
        matches,
        used,
        counter,
        x_subdivisions,
        key_count_length,
        max_index,
    }

    public AABBCollide_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.aabb_collide));
    }
}
