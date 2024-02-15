package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class PrepareTransforms_k extends GPUKernel<PrepareTransforms_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        transforms(Sizeof.cl_mem),
        hull_rotations(Sizeof.cl_mem),
        indices(Sizeof.cl_mem),
        transforms_out(Sizeof.cl_mem),
        offset(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public PrepareTransforms_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.prepare_transforms.gpu.kernels().get(GPU.Kernel.prepare_transforms), Args.values());
    }
}
