package com.controllerface.bvge.cl.buffers;

public class BasicBufferGroup extends BufferGroup<CoreBufferType>
{
    public BasicBufferGroup(String name, long ptr_queue)
    {
        super(CoreBufferType.class, name, ptr_queue);
    }
}
