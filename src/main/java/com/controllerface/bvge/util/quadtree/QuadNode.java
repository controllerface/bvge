package com.controllerface.bvge.util.quadtree;

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