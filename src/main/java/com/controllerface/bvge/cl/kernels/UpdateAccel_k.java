package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class UpdateAccel_k extends GPUKernel<UpdateAccel_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        armature_accel(Sizeof.cl_mem),
        target(Sizeof.cl_int),
        new_value(Sizeof.cl_float2);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public UpdateAccel_k(cl_command_queue command_queue)
    {
        super(command_queue, GPU.Program.gpu_crud.gpu.kernels().get(GPU.Kernel.update_accel), Args.values());
    }
}
