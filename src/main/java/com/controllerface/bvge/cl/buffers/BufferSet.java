package com.controllerface.bvge.cl.buffers;

public interface BufferSet<E extends Enum<E> & BufferType> extends Destoryable
{
    ResizableBuffer get_buffer(E bufferType);
    void set_buffer(E bufferType);
    void set_buffer(E bufferType, long initial_capacity);
    void destroy();
}
