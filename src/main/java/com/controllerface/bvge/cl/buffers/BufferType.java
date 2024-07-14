package com.controllerface.bvge.cl.buffers;

import com.controllerface.bvge.cl.CLData;

public interface BufferType
{
    /**
     * Indicates the size of one buffer element. This is intended to be a static value, it
     * should never change at runtime.
     *
     * @return size in bytes of a single element of the specified buffer type
     */
    CLData.CLType data_type();
}
