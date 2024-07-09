package com.controllerface.bvge.ecs.components;

public final class Position implements GameComponent
{
    public float x;
    public float y;

    public Position(float x, float y)
    {
        this.x = x;
        this.y = y;
    }
}
