package com.controllerface.bvge.gpu.cl.buffers;

import com.controllerface.bvge.gpu.GPUResource;

public interface BufferSet<E extends Enum<E> & BufferType> extends GPUResource
{
    ResizableBuffer buffer(E bufferType);
    void init_buffer(E bufferType);
    void init_buffer(E bufferType, long initial_capacity);
    void release();
}
