package com.controllerface.bvge.memory.types;

import com.controllerface.bvge.gpu.cl.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.buffers.BufferType;

import static com.controllerface.bvge.gpu.cl.CL_DataTypes.cl_int;

public enum CollectedBufferType implements BufferType
{
    TYPES(cl_int),

    ;

    private final CL_DataTypes.CL_Type data_type;

    CollectedBufferType(CL_DataTypes.CL_Type itemSize)
    {
        data_type = itemSize;
    }

    @Override
    public CL_DataTypes.CL_Type data_type()
    {
        return data_type;
    }
}
