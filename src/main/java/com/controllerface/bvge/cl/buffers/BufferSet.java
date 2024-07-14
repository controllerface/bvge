package com.controllerface.bvge.cl.buffers;

public interface BufferSet<E extends Enum<E> & BufferType> extends Destroyable
{
    ResizableBuffer buffer(E bufferType);
    void init_buffer(E bufferType);
    void init_buffer(E bufferType, long initial_capacity);
    void destroy();
}
