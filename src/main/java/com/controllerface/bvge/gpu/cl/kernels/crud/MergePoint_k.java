package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class MergePoint_k extends GPUKernel
{
    public enum Args
    {
        points_in,
        point_vertex_references_in,
        point_hull_indices_in,
        point_hit_counts_in,
        point_flags_in,
        point_bone_tables_in,
        points_out,
        point_vertex_references_out,
        point_hull_indices_out,
        point_hit_counts_out,
        point_flags_out,
        point_bone_tables_out,
        point_offset,
        bone_offset,
        hull_offset,
        max_point,
    }

    public MergePoint_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.merge_point));
    }
}
