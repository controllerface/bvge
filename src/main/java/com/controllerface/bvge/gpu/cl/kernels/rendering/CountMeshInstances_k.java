package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.RenderBufferType.*;

public class CountMeshInstances_k extends GPUKernel
{
    public enum Args
    {
        hull_mesh_ids,
        hull_flags,
        hull_entity_ids,
        entity_flags,
        counters,
        query,
        total,
        count,
        max_hull,
    }

    public CountMeshInstances_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.count_mesh_instances));
    }

    public GPUKernel init(CL_Buffer counter_buf,
                          CL_Buffer query_buf,
                          CL_Buffer totals_buf,
                          int mesh_count)
    {
        return this.buf_arg(Args.counters, counter_buf)
            .buf_arg(Args.query, query_buf)
            .buf_arg(Args.total, totals_buf)
            .set_arg(Args.count, mesh_count)
            .buf_arg(Args.hull_mesh_ids, GPU.memory.get_buffer(RENDER_HULL_MESH_ID))
            .buf_arg(Args.hull_flags, GPU.memory.get_buffer(RENDER_HULL_FLAG))
            .buf_arg(Args.hull_entity_ids, GPU.memory.get_buffer(RENDER_HULL_ENTITY_ID))
            .buf_arg(Args.entity_flags, GPU.memory.get_buffer(RENDER_ENTITY_FLAG));
    }
}
