package com.controllerface.bvge.game;

import com.controllerface.bvge.ecs.ECS;

public abstract class GameMode
{
    protected final ECS ecs;

    public GameMode(ECS ecs)
    {
        this.ecs = ecs;
    }

    abstract public void start();

    abstract public void update(float dt);

    abstract public void resizeSpatialMap(int width, int height);

    abstract public void load();
}