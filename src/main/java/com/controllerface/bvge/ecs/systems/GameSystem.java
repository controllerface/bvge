package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.ecs.ECS;

/**
 * Base class for all game systems. Implementations will typically be registered to the provided
 * ECS object, which ensure the tick() method is called every time the game loop executes. Systems
 * will always run in a predefined order, determined by the order in which they are registered by
 * calling code.
 */
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
