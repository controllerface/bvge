package com.controllerface.bvge.memory.types;

import com.controllerface.bvge.gpu.cl.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.buffers.BufferType;

import static com.controllerface.bvge.gpu.cl.CL_DataTypes.cl_float2;
import static com.controllerface.bvge.gpu.cl.CL_DataTypes.cl_int;

public enum BrokenBufferType implements BufferType
{
    BROKEN_POSITIONS(cl_float2),
    BROKEN_ENTITY_TYPES(cl_int),
    BROKEN_MODEL_IDS(cl_int),

    ;

    private final CL_DataTypes.CL_Type data_type;

    BrokenBufferType(CL_DataTypes.CL_Type itemSize)
    {
        data_type = itemSize;
    }

    @Override
    public CL_DataTypes.CL_Type data_type()
    {
        return data_type;
    }
}
