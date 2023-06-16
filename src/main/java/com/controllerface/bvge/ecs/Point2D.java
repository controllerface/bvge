package com.controllerface.bvge.ecs;

import org.joml.Vector2f;

public class Point2D
{
    private Vector2f pos;
    private Vector2f prv;

    public Point2D(Vector2f pos)
    {
        this.pos = pos;
        this.prv = new Vector2f(pos.x, pos.y);
    }

    public Vector2f getPos()
    {
        return pos;
    }

    public Vector2f getPrv()
    {
        return prv;
    }
}
