package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class PrepareEdges_k extends GPUKernel
{
    public enum Args
    {
        points,
        point_hull_indices,
        hull_flags,
        edges,
        edge_flags,
        vertex_vbo,
        flag_vbo,
        offset,
        max_edge,
    }

    public PrepareEdges_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.prepare_edges));
    }
}
