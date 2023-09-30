package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.ecs.ECS;

public abstract class GameSystem
{
    protected final ECS ecs;

    public GameSystem(ECS ecs)
    {
        this.ecs = ecs;
    }

    abstract public void tick(float dt);

    public void shutdown()
    {
        // todo: some systems probably should implement this in final code, or if unneeded,
        //  it should be removed
    }
}
