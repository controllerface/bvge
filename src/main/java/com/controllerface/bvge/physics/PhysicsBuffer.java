package com.controllerface.bvge.physics;

import com.controllerface.bvge.cl.buffers.BufferType;

import static com.controllerface.bvge.cl.CLData.*;

public enum PhysicsBuffer implements BufferType
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

    private final CLType item_size;

    PhysicsBuffer(CLType itemSize)
    {
        item_size = itemSize;
    }

    @Override
    public CLType data_type()
    {
        return item_size;
    }
}
