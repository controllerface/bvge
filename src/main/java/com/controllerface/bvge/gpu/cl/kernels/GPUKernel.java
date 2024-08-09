package com.controllerface.bvge.gpu.cl.kernels;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public abstract class GPUKernel
{
    final CL_CommandQueue cmd_queue;
    final CL_Kernel kernel;
    final List<CL_Buffer> shared_memory_ptrs = new ArrayList<>();

    public GPUKernel(CL_CommandQueue cmd_queue, CL_Kernel kernel)
    {
        this.cmd_queue = cmd_queue;
        this.kernel = kernel;
    }

    public GPUKernel share_mem(CL_Buffer mem_ptr)
    {
        shared_memory_ptrs.add(mem_ptr);
        return this;
    }

    public GPUKernel buf_arg(Enum<?> arg, ResizableBuffer buffer)
    {
        buffer.register(this, arg);
        kernel.ptr_arg(arg.ordinal(), buffer.pointer());
        return this;
    }

    public GPUKernel buf_arg(Enum<?> arg, CL_Buffer buffer)
    {
        kernel.ptr_arg(arg.ordinal(), buffer.ptr());
        return this;
    }

    public GPUKernel ptr_arg(Enum<?> arg, long pointer)
    {
        kernel.ptr_arg(arg.ordinal(), pointer);
        return this;
    }

    public GPUKernel ptr_arg(Enum<?> arg, ByteBuffer buffer)
    {
        kernel.ptr_arg(arg.ordinal(), buffer);
        return this;
    }

    public GPUKernel loc_arg(Enum<?> arg, long size)
    {
        kernel.loc_arg(arg.ordinal(), size);
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, double[] value)
    {
        kernel.set_arg(arg.ordinal(), value);
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, double value)
    {
        kernel.set_arg(arg.ordinal(), value);
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, float[] value)
    {
        kernel.set_arg(arg.ordinal(), value);
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, float value)
    {
        kernel.set_arg(arg.ordinal(), value);
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, short[] value)
    {
        kernel.set_arg(arg.ordinal(), value);
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, short value)
    {
        kernel.set_arg(arg.ordinal(), value);
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, int[] value)
    {
        kernel.set_arg(arg.ordinal(), value);
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, int value)
    {
        kernel.set_arg(arg.ordinal(), value);
        return this;
    }

    public void call_task()
    {
        call(GPU.compute.global_single_size, GPU.compute.global_single_size);
    }

    public void call(long[] global_work_size, long[] local_work_size)
    {
        call(global_work_size, local_work_size, null);
    }

    public void call(long[] global_work_size, long[] local_work_size, long[] global_work_offset)
    {
        if (!shared_memory_ptrs.isEmpty())
        {
            GPU.CL.gl_acquire(cmd_queue, shared_memory_ptrs);
        }

        GPU.CL.kernel_call(cmd_queue, kernel, global_work_size, local_work_size, global_work_offset);

        if (!shared_memory_ptrs.isEmpty())
        {
            GPU.CL.gl_release(cmd_queue, shared_memory_ptrs);
        }

        shared_memory_ptrs.clear();
    }
}
