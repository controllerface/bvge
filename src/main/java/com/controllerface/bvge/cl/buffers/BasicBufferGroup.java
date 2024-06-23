package com.controllerface.bvge.cl.buffers;

public class BasicBufferGroup extends BufferGroup
{
    public BasicBufferGroup(String name, long ptr_queue)
    {
        super(CoreBufferType.class, name, ptr_queue);
    }
}
