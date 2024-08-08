package com.controllerface.bvge.ecs.components;

import com.controllerface.bvge.ecs.GameComponent;

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
