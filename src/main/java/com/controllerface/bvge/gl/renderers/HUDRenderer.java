package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;

public class HUDRenderer extends GameSystem
{

    public HUDRenderer(ECS ecs)
    {
        super(ecs);
    }

    @Override
    public void tick(float dt)
    {

    }

    @Override
    public void shutdown()
    {
        super.shutdown();
    }
}