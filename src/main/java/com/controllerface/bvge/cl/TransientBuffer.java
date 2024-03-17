package com.controllerface.bvge.cl;

import java.util.ArrayList;
import java.util.List;

public class TransientBuffer implements ResizableBuffer
{
    private static final long DEFAULT_CAPACITY = 256L;

    private final int item_size;

    private long capacity;
    private long pointer;

    private final List<RegisteredKernel> registered_kernels = new ArrayList<>();

    private record RegisteredKernel(GPUKernel kernel, Enum<?> arg) { }

    public TransientBuffer(int item_size)
    {
        this.item_size = item_size;
        this.capacity = this.item_size * DEFAULT_CAPACITY;
        this.pointer = GPGPU.cl_new_buffer(this.capacity);
    }

    private void reset_kernels()
    {
        for (var reg : registered_kernels)
        {
            reg.kernel.ptr_arg(reg.arg, this.pointer);
        }
    }

    private void ensure_size(long size_bytes)
    {
        if (size_bytes <= this.capacity) return;
        release();
        this.capacity = size_bytes;
        this.pointer = GPGPU.cl_new_buffer(this.capacity);
        reset_kernels();
    }

    @Override
    public void ensure_capacity(long capacity)
    {
        ensure_size(item_size * capacity);
    }

    @Override
    public long pointer()
    {
        return pointer;
    }

    @Override
    public void register(GPUKernel kernel, Enum<?> arg)
    {
        registered_kernels.add(new RegisteredKernel(kernel, arg));
    }


    @Override
    public void clear()
    {
        GPGPU.cl_zero_buffer(this.pointer, this.capacity);
    }

    @Override
    public void release()
    {
        GPGPU.cl_release_buffer(this.pointer);
    }
}
