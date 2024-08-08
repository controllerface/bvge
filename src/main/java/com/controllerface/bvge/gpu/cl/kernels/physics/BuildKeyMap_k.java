package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class BuildKeyMap_k extends GPUKernel
{
    public enum Args
    {
        hull_aabb_index,
        hull_aabb_key_table,
        key_map,
        key_offsets,
        key_counts,
        x_subdivisions,
        key_count_length,
        max_hull,
    }

    public BuildKeyMap_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.build_key_map));
    }
}
