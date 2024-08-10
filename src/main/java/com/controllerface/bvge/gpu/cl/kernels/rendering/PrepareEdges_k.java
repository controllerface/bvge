package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.RenderBufferType.*;

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

    public GPUKernel init(CL_Buffer vbo_edge, CL_Buffer vbo_flag)
    {
        return this.buf_arg(PrepareEdges_k.Args.vertex_vbo, vbo_edge)
            .buf_arg(PrepareEdges_k.Args.flag_vbo, vbo_flag)
            .buf_arg(PrepareEdges_k.Args.points, GPU.memory.get_buffer(RENDER_POINT))
            .buf_arg(PrepareEdges_k.Args.point_hull_indices, GPU.memory.get_buffer(RENDER_POINT_HULL_INDEX))
            .buf_arg(PrepareEdges_k.Args.hull_flags, GPU.memory.get_buffer(RENDER_HULL_FLAG))
            .buf_arg(PrepareEdges_k.Args.edges, GPU.memory.get_buffer(RENDER_EDGE))
            .buf_arg(PrepareEdges_k.Args.edge_flags, GPU.memory.get_buffer(RENDER_EDGE_FLAG));
    }
}
