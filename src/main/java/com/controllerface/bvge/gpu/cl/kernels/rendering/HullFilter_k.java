package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.RenderBufferType.RENDER_HULL_MESH_ID;

public class HullFilter_k extends GPUKernel
{
    public enum Args
    {
        hull_mesh_ids,
        hulls_out,
        counter,
        mesh_id,
        max_hull,
    }

    public HullFilter_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.hull_filter));
    }

    public GPUKernel init()
    {
        return this.buf_arg(Args.hull_mesh_ids, GPU.memory.get_buffer(RENDER_HULL_MESH_ID));
    }
}
