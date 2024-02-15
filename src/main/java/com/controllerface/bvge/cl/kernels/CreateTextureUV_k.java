package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CreateTextureUV_k extends GPUKernel<CreateTextureUV_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        texture_uvs(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_texture_uv(Sizeof.cl_float2);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CreateTextureUV_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.gpu_crud.gpu.kernels().get(GPU.Kernel.create_texture_uv), Args.values());
    }
}
