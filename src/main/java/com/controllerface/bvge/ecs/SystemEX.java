package com.controllerface.bvge.ecs;

public abstract class SystemEX
{
    protected final ECS ecs;

    public SystemEX(ECS ecs)
    {
        this.ecs = ecs;
    }

    abstract public void run(float dt);
}
