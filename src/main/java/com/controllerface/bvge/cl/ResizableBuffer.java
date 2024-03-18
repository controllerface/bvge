package com.controllerface.bvge.cl;

public abstract class ResizableBuffer
{
    private static final long DEFAULT_ITEM_CAPACITY = 256L;

    private final int item_size;

    protected long item_capacity;
    protected long byte_capacity;
    protected long used_item_capacity;
    protected long pointer;

    private RegisteredKernel[] registered_kernels = new RegisteredKernel[0];

    private record RegisteredKernel(GPUKernel kernel, Enum<?> arg) { }

    public ResizableBuffer(int item_size)
    {
        this.item_size = item_size;
        this.item_capacity = DEFAULT_ITEM_CAPACITY;
        this.byte_capacity = this.item_size * this.item_capacity;
        this.used_item_capacity = 0;
        this.pointer = GPGPU.cl_new_buffer(this.byte_capacity);
    }

    abstract long resize(long old_size);

    private void update_registered_kernels()
    {
        for (var registered : registered_kernels)
        {
            registered.kernel.ptr_arg(registered.arg, this.pointer);
        }
    }

    public void ensure_total_capacity(long total_item_capacity)
    {
        var byte_capacity = item_size * total_item_capacity;
        if (byte_capacity <= this.byte_capacity) return;
        long previous_capacity = this.byte_capacity;
        this.byte_capacity = byte_capacity;
        this.item_capacity = total_item_capacity;
        this.used_item_capacity = total_item_capacity;
        this.pointer = resize(previous_capacity);
        update_registered_kernels();
    }

    public void ensure_partial_capacity(long partial_item_capacity)
    {
        this.used_item_capacity += partial_item_capacity;
        if (this.used_item_capacity > this.item_capacity)
        {
            long previous_capacity = this.byte_capacity;
            while (this.item_capacity < this.used_item_capacity)
            {
                this.item_capacity += DEFAULT_ITEM_CAPACITY;
            }
            this.byte_capacity = this.item_size * this.item_capacity;
            this.pointer = resize(previous_capacity);
            update_registered_kernels();
        }
    }

    public void release_item_capacity(long released_item_capacity)
    {
        this.used_item_capacity -= released_item_capacity;
        assert this.used_item_capacity >= 0 : "negative used capacity error. check buffer usage logic";
    }

    public long pointer()
    {
        return pointer;
    }

    public void register(GPUKernel kernel, Enum<?> arg)
    {
        var new_copy = new RegisteredKernel[registered_kernels.length + 1];
        System.arraycopy(registered_kernels, 0, new_copy, 0, registered_kernels.length);
        new_copy[new_copy.length - 1] = new RegisteredKernel(kernel, arg);
        registered_kernels = new_copy;
    }

    public void clear()
    {
        GPGPU.cl_zero_buffer(this.pointer, this.byte_capacity);
    }

    public void release()
    {
        GPGPU.cl_release_buffer(this.pointer);
    }
}
