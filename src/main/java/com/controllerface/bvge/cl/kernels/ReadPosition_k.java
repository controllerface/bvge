package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class ReadPosition_k extends GPUKernel<ReadPosition_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        armatures(Sizeof.cl_mem),
        output(Sizeof.cl_float2),
        target(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public ReadPosition_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.gpu_crud.gpu.kernels().get(GPU.Kernel.read_position), Args.values());
    }
}
