package com.controllerface.bvge.game;

import com.controllerface.bvge.ecs.ECS;

public abstract class GameMode
{
    protected final ECS ecs;

    public GameMode(ECS ecs)
    {
        this.ecs = ecs;
    }

    abstract public void init();

    abstract public void destroy();
}