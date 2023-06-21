package com.controllerface.bvge.util.quadtree;

import com.controllerface.bvge.ecs.GameComponent;
import com.controllerface.bvge.ecs.systems.physics.VerletPhysics;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class QuadRectangle implements GameComponent
{

    public float max_x = Float.MIN_VALUE;
    public float min_x = Float.MAX_VALUE;
    public float max_y = Float.MIN_VALUE;
    public float min_y = Float.MAX_VALUE;

    public float x, y, width, height;

    public final boolean playerTouch;

    private Set<VerletPhysics.BoxKey> keys = new HashSet<>();


    public QuadRectangle(float x, float y,
                         float width, float height)
    {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.playerTouch = false;
    }

    public QuadRectangle(float x, float y,
                         float width, float height, boolean playerTouch)
    {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.playerTouch = playerTouch;
    }


    public QuadRectangle(float x, float y,
                         float width, float height,
                         float max_x, float min_x,
                         float max_y, float min_y)
    {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.max_x = max_x;
        this.min_x = min_x;
        this.max_y = max_y;
        this.min_y = min_y;
        this.playerTouch = false;
    }

    public void resetKeys()
    {
        keys.clear();
    }

    public void addkey(VerletPhysics.BoxKey key)
    {
        keys.add(key);

    }

    public Set<VerletPhysics.BoxKey> getKeys()
    {
        return keys;
    }


    public void setX(float x)
    {
        this.x = x;
    }

    public void setMax_x(float max_x)
    {
        this.max_x = max_x;
    }

    public void setMin_x(float min_x)
    {
        this.min_x = min_x;
    }

    public void setMax_y(float max_y)
    {
        this.max_y = max_y;
    }

    public void setMin_y(float min_y)
    {
        this.min_y = min_y;
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