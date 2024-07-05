package com.controllerface.bvge.ecs.components;

import com.controllerface.bvge.substances.Solid;

public class BlockCursor implements GameComponent
{
    private Solid block = null;

    public Solid block()
    {
        return block;
    }

    public boolean is_active()
    {
        return block != null;
    }

    public void set_block(Solid block)
    {
        this.block = block;
    }
}
