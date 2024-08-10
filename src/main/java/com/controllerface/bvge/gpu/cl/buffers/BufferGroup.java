package com.controllerface.bvge.gpu.cl.buffers;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;

public class BufferGroup<E extends Enum<E> & BufferType> implements BufferSet<E>
{
    private final String name;
    private final boolean persistent;
    protected final CL_CommandQueue cmd_queue;
    protected Map<E, ResizableBuffer> buffers;

    public BufferGroup(CL_CommandQueue cmd_queue, Class<E> type, String name, boolean persistent)
    {
        this.buffers = Collections.synchronizedMap(new EnumMap<>(type));
        this.name = name;
        this.cmd_queue = cmd_queue;
        this.persistent = persistent;
    }

    private ResizableBuffer new_buffer(int item_size)
    {
        return new PersistentBuffer(this.cmd_queue, item_size);
    }

    private ResizableBuffer new_buffer(int item_size, long initial_capacity)
    {
        return persistent
            ? new PersistentBuffer(this.cmd_queue, item_size, initial_capacity)
            : new TransientBuffer(this.cmd_queue, item_size, initial_capacity);
    }

    @Override
    public ResizableBuffer buffer(E buffer_type)
    {
        return buffers.get(buffer_type);
    }

    @Override
    public void init_buffer(E buffer_type)
    {
        if (buffers.containsKey(buffer_type))
        {
            throw new RuntimeException("Buffer type: " + buffer_type + " already exists in: " + name);
        }
        buffers.put(buffer_type, new_buffer(buffer_type.data_type().size()));
    }

    @Override
    public void init_buffer(E buffer_type, long initial_capacity)
    {
        buffers.put(buffer_type, new_buffer(buffer_type.data_type().size(), initial_capacity));
    }

    @Override
    public void release()
    {
        long[] total = new long[1];
        buffers.forEach((_, v) ->
        {
            total[0] += v.debug_data();
            v.release();
        });
        System.out.println("BufferGroup [" + name + "] Memory Usage: MB " + String.format("%.4f", (float) total[0] / 1024f / 1024f));
    }
}