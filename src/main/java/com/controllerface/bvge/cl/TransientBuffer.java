package com.controllerface.bvge.cl;

public class TransientBuffer implements ResizableBuffer
{
    private static final long DEFAULT_CAPACITY = 256L;

    private final int item_size;

    private long capacity;
    private long pointer;

    private RegisteredKernel[] registered_kernels = new RegisteredKernel[0];

    private record RegisteredKernel(GPUKernel kernel, Enum<?> arg) { }

    public TransientBuffer(int item_size)
    {
        this.item_size = item_size;
        this.capacity = this.item_size * DEFAULT_CAPACITY;
        this.pointer = GPGPU.cl_new_buffer(this.capacity);
    }

    private void update_registered_kernels()
    {
        for (var registered : registered_kernels)
        {
            registered.kernel.ptr_arg(registered.arg, this.pointer);
        }
    }

    private void ensure_size(long size_bytes)
    {
        if (size_bytes <= this.capacity) return;
        release();
        this.capacity = size_bytes;
        this.pointer = GPGPU.cl_new_buffer(this.capacity);
        update_registered_kernels();
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
        var new_copy = new RegisteredKernel[registered_kernels.length + 1];
        System.arraycopy(registered_kernels, 0, new_copy, 0, registered_kernels.length);
        new_copy[new_copy.length - 1] = new RegisteredKernel(kernel, arg);
        registered_kernels = new_copy;
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
