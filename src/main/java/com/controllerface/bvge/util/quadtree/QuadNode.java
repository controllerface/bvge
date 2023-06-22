package com.controllerface.bvge.util.quadtree;

import com.controllerface.bvge.ecs.components.QuadRectangle;

public class QuadNode<T>
{
    QuadRectangle r;
    T element;

    QuadNode(QuadRectangle r, T element)
    {
        this.r = r;
        this.element = element;
    }

    @Override
    public String toString()
    {
        return r.toString();
    }
}