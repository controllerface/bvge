package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.ecs.ECS;

public abstract class GameSystem
{
    protected final ECS ecs;

    public GameSystem(ECS ecs)
    {
        this.ecs = ecs;
    }

    abstract public void run(float dt);
}
