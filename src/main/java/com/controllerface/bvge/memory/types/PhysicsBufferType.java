package com.controllerface.bvge.memory.types;

import com.controllerface.bvge.gpu.cl.buffers.BufferType;

import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.*;

public enum PhysicsBufferType implements BufferType
{
    POINT_REACTION_COUNTS(cl_int),
    POINT_REACTION_OFFSETS(cl_int),
    REACTIONS_IN(cl_float8),
    REACTIONS_OUT(cl_float8),
    REACTION_INDEX(cl_int),
    KEY_MAP(cl_int),
    KEY_BANK(cl_int),
    IN_BOUNDS(cl_int),
    CANDIDATES(cl_int2),
    CANDIDATE_COUNTS(cl_int2),
    CANDIDATE_OFFSETS(cl_int),
    MATCHES(cl_int),
    MATCHES_USED(cl_int),

    ;

    private final CL_Type item_size;

    PhysicsBufferType(CL_Type itemSize)
    {
        item_size = itemSize;
    }

    @Override
    public CL_Type data_type()
    {
        return item_size;
    }
}
