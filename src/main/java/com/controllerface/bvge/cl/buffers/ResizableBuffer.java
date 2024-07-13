package com.controllerface.bvge.cl.buffers;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.cl.kernels.GPUKernel;

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

    public void copy_from(ResizableBuffer source)
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

    public void clear_negative()
    {
        GPGPU.cl_negative_one_buffer(this.queue_pointer, this.buffer_pointer, this.byte_capacity);
    }

    public void release()
    {
        GPGPU.cl_release_buffer(this.buffer_pointer);
    }

    public void transfer_out_int(int[] ints, long size, int count)
    {
        GPGPU.cl_map_read_int_buffer(this.queue_pointer, this.buffer_pointer, size, count, ints);
    }

    public void transfer_out_float(float[] floats, long size, int count)
    {
        GPGPU.cl_map_read_float_buffer(this.queue_pointer, this.buffer_pointer, size, count, floats);
    }

    public void transfer_out_short(short[] shorts, long size, int count)
    {
        GPGPU.cl_map_read_short_buffer(this.queue_pointer, this.buffer_pointer, size, count, shorts);
    }

    public long debug_data()
    {
        return this.byte_capacity;
    }
}
