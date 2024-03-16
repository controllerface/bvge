package com.controllerface.bvge.cl;

import org.lwjgl.system.MemoryStack;

import java.util.ArrayList;
import java.util.List;

import static com.controllerface.bvge.cl.CLUtils.k_call;
import static org.lwjgl.opencl.CL12.clSetKernelArg;

public abstract class GPUKernel
{
    final long command_queue_ptr;
    final long kernel_ptr;
    final List<Long> shared_memory_ptrs = new ArrayList<>();

    public GPUKernel(long command_queue_ptr, long kernel_ptr)
    {
        this.command_queue_ptr = command_queue_ptr;
        this.kernel_ptr = kernel_ptr;
    }

    public GPUKernel share_mem(long mem_ptr)
    {
        shared_memory_ptrs.add(mem_ptr);
        return this;
    }

    public GPUKernel buf_arg(Enum<?> val, ResizableBuffer buffer)
    {
        buffer.register(this, val);
        return ptr_arg(val.ordinal(), buffer.pointer());
    }

    public GPUKernel mem_arg(Enum<?> val, GPUMemory gpu_memory)
    {
        return ptr_arg(val.ordinal(), gpu_memory.pointer());
    }

    public GPUKernel ptr_arg(Enum<?> val, long pointer)
    {
        return ptr_arg(val.ordinal(), pointer);
    }

    public GPUKernel loc_arg(Enum<?> pos, long size)
    {
        clSetKernelArg(this.kernel_ptr, pos.ordinal(), size);
        return this;
    }

    public GPUKernel set_arg(Enum<?> pos, double[] value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var doubleBuffer = mem_stack.doubles(value);
            clSetKernelArg(this.kernel_ptr, pos.ordinal(), doubleBuffer);
        }
        return this;
    }

    public GPUKernel set_arg(Enum<?> pos, double value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var doubleBuffer = mem_stack.doubles(value);
            clSetKernelArg(this.kernel_ptr, pos.ordinal(), doubleBuffer);
        }
        return this;
    }

    public GPUKernel set_arg(Enum<?> pos, float[] value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var floatBuffer = mem_stack.floats(value);
            clSetKernelArg(this.kernel_ptr, pos.ordinal(), floatBuffer);
        }
        return this;
    }

    public GPUKernel set_arg(Enum<?> pos, float value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var floatBuffer = mem_stack.floats(value);
            clSetKernelArg(this.kernel_ptr, pos.ordinal(), floatBuffer);
        }
        return this;
    }

    public GPUKernel set_arg(Enum<?> pos, int[] value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var intBuffer = mem_stack.ints(value);
            clSetKernelArg(this.kernel_ptr, pos.ordinal(), intBuffer);
        }
        return this;
    }

    public GPUKernel set_arg(Enum<?> pos, int value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var intBuffer = mem_stack.ints(value);
            clSetKernelArg(this.kernel_ptr, pos.ordinal(), intBuffer);
        }
        return this;
    }

    private GPUKernel ptr_arg(int pos, long pointer)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var pointerBuffer = mem_stack.callocPointer(1).put(0, pointer);
            clSetKernelArg(this.kernel_ptr, pos, pointerBuffer);
        }
        return this;
    }

    public void call(long[] global_work_size)
    {
        call(global_work_size, null);
    }

    public void call(long[] global_work_size, long[] local_work_size)
    {
        call(global_work_size, local_work_size, null);
    }

    public void call(long[] global_work_size, long[] local_work_size, long[] global_work_offset)
    {
        if (!shared_memory_ptrs.isEmpty())
        {
            CLUtils.gl_acquire(command_queue_ptr, shared_memory_ptrs);
        }

        k_call(command_queue_ptr, kernel_ptr, global_work_size, local_work_size, global_work_offset);

        if (!shared_memory_ptrs.isEmpty())
        {
            CLUtils.gl_release(command_queue_ptr, shared_memory_ptrs);
        }

        shared_memory_ptrs.clear();
    }
}
