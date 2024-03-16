package com.controllerface.bvge.cl;

public interface ResizableBuffer
{
    void ensure_capacity(long capacity);

    long pointer();

    void release();

    void register(GPUKernel kernel, Enum<?> arg);
}
