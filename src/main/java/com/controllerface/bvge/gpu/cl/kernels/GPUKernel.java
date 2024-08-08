package com.controllerface.bvge.gpu.cl.kernels;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import org.lwjgl.system.MemoryStack;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static com.controllerface.bvge.gpu.cl.CLUtils.k_call;
import static org.lwjgl.opencl.CL12.clSetKernelArg;
import static org.lwjgl.opencl.CL20.clSetKernelArgSVMPointer;

public abstract class GPUKernel
{
    final CL_CommandQueue cmd_queue;
    final long kernel_ptr;
    final List<CL_Buffer> shared_memory_ptrs = new ArrayList<>();

    public GPUKernel(CL_CommandQueue cmd_queue, long kernel_ptr)
    {
        this.cmd_queue = cmd_queue;
        this.kernel_ptr = kernel_ptr;
    }

    public GPUKernel share_mem(CL_Buffer mem_ptr)
    {
        shared_memory_ptrs.add(mem_ptr);
        return this;
    }

    public GPUKernel buf_arg(Enum<?> arg, ResizableBuffer buffer)
    {
        buffer.register(this, arg);
        return ptr_arg(arg, buffer.pointer());
    }

    public GPUKernel buf_arg(Enum<?> arg, CL_Buffer buffer)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var pointerBuffer = mem_stack.callocPointer(1).put(0, buffer.ptr());
            clSetKernelArg(this.kernel_ptr, arg.ordinal(), pointerBuffer);
        }
        return this;
    }

    public GPUKernel ptr_arg(Enum<?> arg, long pointer)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var pointerBuffer = mem_stack.callocPointer(1).put(0, pointer);
            clSetKernelArg(this.kernel_ptr, arg.ordinal(), pointerBuffer);
        }
        return this;
    }

    public GPUKernel ptr_arg(Enum<?> arg, ByteBuffer buffer)
    {
        clSetKernelArgSVMPointer(this.kernel_ptr, arg.ordinal(), buffer);
        return this;
    }

    public GPUKernel loc_arg(Enum<?> arg, long size)
    {
        clSetKernelArg(this.kernel_ptr, arg.ordinal(), size);
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, double[] value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var doubleBuffer = mem_stack.doubles(value);
            clSetKernelArg(this.kernel_ptr, arg.ordinal(), doubleBuffer);
        }
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, double value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var doubleBuffer = mem_stack.doubles(value);
            clSetKernelArg(this.kernel_ptr, arg.ordinal(), doubleBuffer);
        }
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, float[] value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var floatBuffer = mem_stack.floats(value);
            clSetKernelArg(this.kernel_ptr, arg.ordinal(), floatBuffer);
        }
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, float value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var floatBuffer = mem_stack.floats(value);
            clSetKernelArg(this.kernel_ptr, arg.ordinal(), floatBuffer);
        }
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, short[] value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var shortBuffer = mem_stack.shorts(value);
            clSetKernelArg(this.kernel_ptr, arg.ordinal(), shortBuffer);
        }
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, short value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var intBuffer = mem_stack.shorts(value);
            clSetKernelArg(this.kernel_ptr, arg.ordinal(), intBuffer);
        }
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, int[] value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var intBuffer = mem_stack.ints(value);
            clSetKernelArg(this.kernel_ptr, arg.ordinal(), intBuffer);
        }
        return this;
    }

    public GPUKernel set_arg(Enum<?> arg, int value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var intBuffer = mem_stack.ints(value);
            clSetKernelArg(this.kernel_ptr, arg.ordinal(), intBuffer);
        }
        return this;
    }

    public void call_task()
    {
        call(GPGPU.compute.global_single_size, GPGPU.compute.global_single_size);
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

        k_call(cmd_queue.ptr(), kernel_ptr, global_work_size, local_work_size, global_work_offset);

        if (!shared_memory_ptrs.isEmpty())
        {
            GPU.CL.gl_release(cmd_queue, shared_memory_ptrs);
        }

        shared_memory_ptrs.clear();
    }
}
