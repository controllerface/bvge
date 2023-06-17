package com.controllerface.bvge.ecs;

import org.joml.Vector2f;

public record Edge2D (Point2D p1, Point2D p2, float length)
{
    public Edge2D(Point2D p1, Point2D p2)
    {
        this(p1, p2, p2.pos().sub(p1.pos(), new Vector2f()).length());
    }
}
