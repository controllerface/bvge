package com.controllerface.bvge.cl.buffers;

import java.util.EnumMap;
import java.util.Map;

public abstract class BufferGroup
{
    protected final long ptr_queue;
    protected Map<BufferType, ResizableBuffer> buffers = new EnumMap<>(BufferType.class);

    public BufferGroup(long ptr_queue)
    {
        this.ptr_queue = ptr_queue;
    }

    public ResizableBuffer buffer(BufferType bufferType)
    {
        return buffers.get(bufferType);
    }

    protected ResizableBuffer new_buffer(int size, long initial_capacity)
    {
        return new PersistentBuffer(this.ptr_queue, size, initial_capacity);
    }

    public void set_buffer(BufferType bufferType, ResizableBuffer resizableBuffer)
    {
        buffers.put(bufferType, resizableBuffer);
    }

    public void destroy()
    {
        long[] total = new long[1];
        buffers.forEach((_, v) ->
        {
            total[0]+= v.debug_data();
            v.release();
        });
        System.out.println("Buffer Group Memory Usage: MB " + ((float) total[0] / 1024f / 1024f));
    }
}
