package com.controllerface.bvge.gpu.cl.buffers;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public abstract class ResizableBuffer
{
    protected static final long DEFAULT_ITEM_CAPACITY = 1024L;

    protected final int item_size;
    protected final CL_CommandQueue cmd_queue;

    protected long byte_capacity;
    protected CL_Buffer buffer;

    private RegisteredKernel[] registered_kernels = new RegisteredKernel[0];

    private record RegisteredKernel(GPUKernel kernel, Enum<?> arg) { }

    public ResizableBuffer(CL_CommandQueue cmd_queue, int item_size, long item_capacity)
    {
        this.item_size = item_size;
        this.cmd_queue = cmd_queue;
        this.byte_capacity = this.item_size * item_capacity;
        this.buffer = GPU.CL.new_buffer(GPGPU.compute.context, this.byte_capacity);
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
            registered.kernel.buf_arg(registered.arg, this.buffer);
        }
    }

    public long pointer()
    {
        return buffer.ptr();
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
        GPU.CL.zero_buffer(this.cmd_queue, this.buffer, this.byte_capacity);
    }

    public void clear_negative()
    {
        GPU.CL.negative_one_buffer(this.cmd_queue, this.buffer, this.byte_capacity);
    }

    public void release()
    {
        this.buffer.release();
    }

    public void transfer_out_int(int[] ints, long size, int count)
    {
        GPGPU.cl_map_read_int_buffer(this.cmd_queue.ptr(), this.buffer.ptr(), size, count, ints);
    }

    public void transfer_out_float(float[] floats, long size, int count)
    {
        GPGPU.cl_map_read_float_buffer(this.cmd_queue.ptr(), this.buffer.ptr(), size, count, floats);
    }

    public void transfer_out_short(short[] shorts, long size, int count)
    {
        GPGPU.cl_map_read_short_buffer(this.cmd_queue.ptr(), this.buffer.ptr(), size, count, shorts);
    }

    public long debug_data()
    {
        return this.byte_capacity;
    }
}
