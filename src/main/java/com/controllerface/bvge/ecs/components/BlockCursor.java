package com.controllerface.bvge.ecs.components;

import com.controllerface.bvge.substances.Solid;

public class BlockCursor implements GameComponent
{
    private Solid block = null;
    private boolean require_unlatch;

    public Solid block()
    {
        return block;
    }

    public boolean is_active()
    {
        return block != null;
    }

    public boolean requires_unlatch()
    {
        return require_unlatch;
    }

    public void set_block(Solid block)
    {
        this.block = block;
    }

    public void set_require_unlatch(boolean require_unlatch)
    {
        this.require_unlatch = require_unlatch;
    }
}
