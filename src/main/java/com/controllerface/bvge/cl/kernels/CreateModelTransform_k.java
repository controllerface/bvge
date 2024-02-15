package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateModelTransform_k extends GPUKernel<CreateModelTransform_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        model_transforms(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_model_transform(Sizeof.cl_float16);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CreateModelTransform_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.gpu_crud.gpu.kernels().get(GPU.Kernel.create_model_transform), Args.values());
    }
}
