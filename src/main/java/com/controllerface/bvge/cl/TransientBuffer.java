package com.controllerface.bvge.cl;

import java.util.ArrayList;
import java.util.List;

public class TransientBuffer implements ResizableBuffer
{
    private static final long DEFAULT_CAPACITY = 1024L; // 1 KB

    private long capacity;
    private long pointer;

    private final List<RegisteredKernel> registered_kernels = new ArrayList<>();

    private record RegisteredKernel(GPUKernel kernel, Enum<?> arg) { }

    public TransientBuffer()
    {
        this.capacity = DEFAULT_CAPACITY;
        this.pointer = GPGPU.cl_new_buffer(this.capacity);
    }

    private void reset_kernels()
    {
        registered_kernels
            .forEach(k->k.kernel.ptr_arg(k.arg, this.pointer));
    }

    @Override
    public void ensure_capacity(long capacity)
    {
        if (capacity <= this.capacity) return;
        release();
        this.capacity = capacity;
        this.pointer = GPGPU.cl_new_buffer(this.capacity);
        reset_kernels();
    }

    @Override
    public long pointer()
    {
        return pointer;
    }

    @Override
    public void release()
    {
        GPGPU.cl_release_buffer(this.pointer);
    }

    @Override
    public void register(GPUKernel kernel, Enum<?> arg)
    {
        registered_kernels.add(new RegisteredKernel(kernel, arg));
    }
}
