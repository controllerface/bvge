package com.controllerface.bvge.gpu.cl.buffers;

public interface BufferType
{
    /**
     * Indicates the size of one buffer element. This is intended to be a static value, it
     * should never change at runtime.
     *
     * @return size in bytes of a single element of the specified buffer type
     */
    CL_DataTypes.CL_Type data_type();
}
