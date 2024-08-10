package com.controllerface.bvge.gpu.cl.kernels.rendering;

import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class CountMeshBatches_k extends GPUKernel
{
    public enum Args
    {
        mesh_details,
        total,
        max_per_batch,
        count,
    }

    public CountMeshBatches_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.count_mesh_batches));
    }

    public GPUKernel init(CL_Buffer total_buf, int max_batch_size)
    {
        return this.buf_arg(Args.total, total_buf)
            .set_arg(Args.max_per_batch, max_batch_size);
    }
}
