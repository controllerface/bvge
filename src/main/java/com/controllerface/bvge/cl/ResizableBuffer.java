package com.controllerface.bvge.cl;

public abstract class ResizableBuffer
{
    protected static final long DEFAULT_ITEM_CAPACITY = 1024L;

    protected final int item_size;
    protected final long queue_pointer;

    protected long byte_capacity;
    protected long buffer_pointer;

    private RegisteredKernel[] registered_kernels = new RegisteredKernel[0];

    private record RegisteredKernel(GPUKernel kernel, Enum<?> arg) { }

    public ResizableBuffer(long queue_ptr, int item_size, long item_capacity)
    {
        this.item_size = item_size;
        this.queue_pointer = queue_ptr;
        this.byte_capacity = this.item_size * item_capacity;
        this.buffer_pointer = GPGPU.cl_new_buffer(this.byte_capacity);
    }

    abstract public void ensure_capacity(long total_item_capacity);

    public void mirror_buffer(ResizableBuffer source)
    {
        throw new UnsupportedOperationException("this operation is not supported");
    }

    protected void update_registered_kernels()
    {
        for (var registered : registered_kernels)
        {
            registered.kernel.ptr_arg(registered.arg, this.buffer_pointer);
        }
    }

    public long pointer()
    {
        return buffer_pointer;
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
        GPGPU.cl_zero_buffer(this.queue_pointer, this.buffer_pointer, this.byte_capacity);
    }

    public void release()
    {
        GPGPU.cl_release_buffer(this.buffer_pointer);
    }

    public long debug_data()
    {
        //System.out.println("buffer total: KB " + this.byte_capacity / 1024f);
        return this.byte_capacity;
    }
}
