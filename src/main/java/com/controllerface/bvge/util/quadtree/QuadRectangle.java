package com.controllerface.bvge.util.quadtree;

import com.controllerface.bvge.ecs.GameComponent;

public class QuadRectangle implements GameComponent
{
    public float x, y, width, height;

    public QuadRectangle(float x, float y, float width, float height)
    {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    public void setX(float x)
    {
        this.x = x;
    }

    public void setY(float y)
    {
        this.y = y;
    }

    public void setWidth(float width)
    {
        this.width = width;
    }

    public void setHeight(float height)
    {
        this.height = height;
    }

    public boolean contains(QuadRectangle r)
    {
        return this.width > 0 && this.height > 0 && r.width > 0 && r.height > 0
            && r.x >= this.x && r.x + r.width <= this.x + this.width
            && r.y >= this.y && r.y + r.height <= this.y + this.height;
    }

    @Override
    public String toString()
    {
        return "x: " + x + " y: " + y + " w: " + width + " h: " + height;
    }
}