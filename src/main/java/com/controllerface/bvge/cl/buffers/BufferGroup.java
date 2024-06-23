package com.controllerface.bvge.cl.buffers;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;

public abstract class BufferGroup<E extends Enum<E> & BufferCategory> implements BufferSet<E>
{
    private final String name;
    protected final long ptr_queue;
    protected Map<E, ResizableBuffer> buffers;

    public BufferGroup(Class<E> type, String name, long ptr_queue)
    {
        this.buffers = Collections.synchronizedMap(new EnumMap<>(type));
        this.name = name;
        this.ptr_queue = ptr_queue;
    }

    private ResizableBuffer new_buffer(int size)
    {
        return new PersistentBuffer(this.ptr_queue, size);
    }

    private ResizableBuffer new_buffer(int size, long initial_capacity)
    {
        return new PersistentBuffer(this.ptr_queue, size, initial_capacity);
    }

    @Override
    public ResizableBuffer get_buffer(E coreBufferType)
    {
        return buffers.get(coreBufferType);
    }

    @Override
    public void set_buffer(E coreBufferType, int size)
    {
        if (buffers.containsKey(coreBufferType))
        {
            throw new RuntimeException("Buffer type: " + coreBufferType + " already exists in: " + name);
        }
        buffers.put(coreBufferType, new_buffer(size));
    }

    @Override
    public void set_buffer(E coreBufferType, int size, long initial_capacity)
    {
        buffers.put(coreBufferType, new_buffer(size, initial_capacity));
    }

    @Override
    public void destroy()
    {
        long[] total = new long[1];
        buffers.forEach((_, v) ->
        {
            total[0]+= v.debug_data();
            v.release();
        });
        System.out.println("BufferGroup [" + name + "] Memory Usage: MB " + ((float) total[0] / 1024f / 1024f));
    }
}
