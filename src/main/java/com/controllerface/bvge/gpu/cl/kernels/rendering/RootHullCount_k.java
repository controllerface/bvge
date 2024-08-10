package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.RenderBufferType.RENDER_ENTITY_MODEL_ID;

public class RootHullCount_k extends GPUKernel
{
    public enum Args
    {
        entity_model_indices,
        counter,
        model_id,
        max_entity,
    }

    public RootHullCount_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.root_hull_count));
    }

    public GPUKernel init()
    {
        return this.buf_arg(Args.entity_model_indices, GPU.memory.get_buffer(RENDER_ENTITY_MODEL_ID));
    }
}
