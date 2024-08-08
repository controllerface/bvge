package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class UpdateSelectBlock_k extends GPUKernel
{
    public enum Args
    {
        entity_flags,
        hull_uv_offsets,
        entity_hull_tables,
        target,
        new_value;
    }

    public UpdateSelectBlock_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.update_select_block));
    }
}
