package com.controllerface.bvge.cl.buffers;

public interface BufferSet<E extends Enum<E> & BufferCategory>
{
    ResizableBuffer get_buffer(E bufferType);
    void set_buffer(E bufferType, int size);
    void set_buffer(E bufferType, int size, long initial_capacity);
    void destroy();
}
