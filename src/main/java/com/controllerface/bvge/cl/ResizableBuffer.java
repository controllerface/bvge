package com.controllerface.bvge.cl;

public abstract class ResizableBuffer
{
    private static final long DEFAULT_ITEM_CAPACITY = 1024L;

    protected final int item_size;

    protected long byte_capacity;
    protected long pointer;

    private RegisteredKernel[] registered_kernels = new RegisteredKernel[0];

    private record RegisteredKernel(GPUKernel kernel, Enum<?> arg) { }

    public ResizableBuffer(int item_size, long item_capacity)
    {
        this.item_size = item_size;
        this.byte_capacity = this.item_size * item_capacity;
        this.pointer = GPGPU.cl_new_buffer(this.byte_capacity);
    }

    public ResizableBuffer(int item_size)
    {
        this.item_size = item_size;
        this.byte_capacity = this.item_size * DEFAULT_ITEM_CAPACITY;
        this.pointer = GPGPU.cl_new_buffer(this.byte_capacity);
    }

    abstract public void ensure_total_capacity(long total_item_capacity);

    protected void update_registered_kernels()
    {
        for (var registered : registered_kernels)
        {
            registered.kernel.ptr_arg(registered.arg, this.pointer);
        }
    }

    public long pointer()
    {
        return pointer;
    }

    public void register(GPUKernel kernel, Enum<?> arg)
    {
        var new_copy = new RegisteredKernel[registered_kernels.length + 1];
        System.arraycopy(registered_kernels, 0, new_copy, 0, registered_kernels.length);
        new_copy[registered_kernels.length] = new RegisteredKernel(kernel, arg);
        registered_kernels = new_copy;
    }

    public void clear()
    {
        GPGPU.cl_zero_buffer(this.pointer, this.byte_capacity);
    }

    public void release()
    {
        GPGPU.cl_release_buffer(this.pointer);
        //System.out.println("buffer total: KB " + this.byte_capacity / 1024f);
    }
}
